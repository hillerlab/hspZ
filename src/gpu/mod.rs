// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! Host-side orchestration of one Seed + Filter pass — the port of
//! `SeedAndFilter()` in `seed_filter.cu`.
//!
//! The per-batch pipeline: generate seeds (host, or device via
//! `seed_kmers`/`scatter_seeds`), `find_num_hits`, the device-resident count
//! scan (`scan_blocks` + host block-sum walk + `add_block_offsets`),
//! `find_hits`, the warp-coalesced score gate, `find_hsps` (X-drop extension),
//! the done-flag scan and `compress_output`.
//!
//! Where the port deviates from the C++, all behaviour-preserving:
//!
//! * `SeedAndFilter` returns the seed-hit count in `anchors[0].score` as a
//!   sentinel element; this returns it in [`FilterOutput`] instead.
//! * The chunk `lower_bound` walk and the sort/dedup/sort tail run on the
//!   host; the scans themselves are device kernels (rounds 8, 10).
//! * The engine is reference-scoped: `Engine::new` uploads the reference
//!   index and `swap_query` replaces only the query side, so one reference bin
//!   serves many query bins; the pass refuses to run on a never-swapped query
//!   (rounds 21–23).
//! * With the default async staging (rounds 29–30) the kernel chain is
//!   enqueued without per-stage host waits and the seed upload rides a second
//!   stream, overlapping the previous batch's compute.
//!
//! Every stage is timed on the host clock so the phases add up to wall time;
//! kernel stages also carry their CUDA-event duration (`gpu ms`).

pub mod kernels;
pub mod sieve;

use crate::hsp::SegmentPair;
use crate::seed::Shape;
use crate::timing::Phases;
use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig,
};
use kernels::{
    BLOCK_SIZE, HSP_BLOCKS, HSP_THREADS, MAX_BLOCKS, MAX_THREADS, NUM_WARPS, SCAN_BLOCK,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `seed_filter.cu: MAX_HITS_PER_GB`.
const MAX_HITS_PER_GB: u64 = 4_194_304;
/// Mean hits per seed at which a `find_hits` launch moves from one thread per
/// seed to one warp per seed: 8 since 2026-10-27 (16 from round 82). On A100s the
/// warp kernel loses below ~8 hits/seed (hg38 chr21 x mm39 at 2.3: +174% kernel
/// time; chr13 at 6.6: +19%) and wins above it (chr7 at 9.4: -16%; chr1 at 12.9:
/// -35%). At 16 the thread kernel also took skewed launches whose mean sits under
/// 16 while most hits live in long seeds: an Ascaphus scaffold bin x an axolotl
/// chromosome (mean 14.5, median 8, max 88,589) ran -82% kernel time and -37%
/// unit wall on the warp kernel, output identical.
const FIND_HITS_WARP_MIN_DENSITY: u64 = 8;

/// The cutoff in effect: `HSPZ_FIND_HITS_WARP_MIN_DENSITY` when set (a benchmark
/// override; `0` sends every launch to the warp kernel), else the shipped 8.
/// Read once; the note on stderr records an override in the run's report.
fn warp_min_density() -> u64 {
    static CUTOFF: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CUTOFF.get_or_init(|| match std::env::var("HSPZ_FIND_HITS_WARP_MIN_DENSITY") {
        Err(_) => FIND_HITS_WARP_MIN_DENSITY,
        Ok(v) => match v.trim().parse::<u64>() {
            Ok(n) => {
                eprintln!("note: find_hits warp cutoff {n} hits/seed (HSPZ_FIND_HITS_WARP_MIN_DENSITY)");
                n
            }
            Err(_) => {
                eprintln!(
                    "warning: HSPZ_FIND_HITS_WARP_MIN_DENSITY={v:?} is not an integer; using {FIND_HITS_WARP_MIN_DENSITY}"
                );
                FIND_HITS_WARP_MIN_DENSITY
            }
        },
    })
}

#[inline]
fn use_warp_find_hits(num_hits: u32, num_seeds: u32) -> bool {
    cfg!(feature = "find-hits-warp")
        && u64::from(num_hits) >= warp_min_density() * u64::from(num_seeds)
}

/// Least `total` the score gate keeps at `hspthresh` (`gate-int-keep`).
///
/// The gate keeps a hit when `((total as f32 as f64) as i32) >= hspthresh`:
/// production rounds the score through f32 before comparing. Round-to-nearest
/// i32 -> f32 and the saturating f64 -> i32 cast are both monotone, so the kept
/// totals are exactly `total >= keep_threshold(hspthresh)` and one integer
/// compare on the device decides the same. The threshold always exists
/// (`i32::MAX` rounds to 2^31, which saturates back to `i32::MAX`). It equals
/// `hspthresh` for -2^24 < `hspthresh` <= 2^24, where every integer is an f32;
/// past that it can move either way: at 16,777,220 the total 16,777,219 rounds
/// up and is kept, at 16,777,217 the total 16,777,217 rounds down and is not.
/// `find_hsps` keeps its own f32 test on the original `hspthresh`.
fn keep_threshold(hspthresh: i32) -> i32 {
    let keeps = |t: i64| ((t as i32) as f32 as f64) as i32 >= hspthresh;
    // Lower bound over [i32::MIN, i32::MAX]: `hi` always keeps, nothing below `lo` does.
    let (mut lo, mut hi) = (i64::from(i32::MIN), i64::from(i32::MAX));
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if keeps(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo as i32
}

/// Result of one `SeedAndFilter` call.
pub struct FilterOutput {
    pub hsps: Vec<SegmentPair>,
    /// Total seed hits expanded — `anchors[0].score` upstream.
    pub num_hits: u32,
    /// HSPs surviving the score threshold, before dedup.
    pub raw_hsps: u32,
    /// The pre-dedup records themselves, kept only when `Engine::dump_raw` is
    /// set. Comparing these against the reference separates an extension
    /// difference from a dedup difference.
    pub raw: Vec<SegmentPair>,
    /// Entropy-accepted raw HSPs classified by source-seed multiplicity for the
    /// env-gated AL3/AM1 diagnostic. Empty on the production path.
    pub audit: Vec<crate::census::AcceptedHsp>,
    /// `--paired` only: this call held more than one pair and its hits exceed
    /// `max_hits`, so it stopped before `find_hits` and produced nothing. The
    /// driver re-issues it one pair per call (`paired.rs`). Always `false`
    /// elsewhere.
    pub oversized: bool,
    /// Round 121: with `defer_dedup`, each `MAX_HITS` chunk's HSPs as they came
    /// off the device, for the caller to put through `hsp::dedup_and_order`
    /// (one chunk at a time, as here) while the next batch's kernels run.
    /// `hsps` is then empty.
    pub chunks: Vec<Vec<SegmentPair>>,
}

/// Round 121: a moment inside one [`Engine::seed_and_filter_with`] call at
/// which kernels are queued and the host has nothing to wait for yet. Whatever
/// the host does between two batches the card spends idle; done here it costs
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lull {
    /// The seed counts and their scan are queued (about 0.1 ms of card time).
    Counts,
    /// The first chunk's hits, score gate and survivor count are queued (the
    /// long stretch of a batch).
    Gate,
}

/// What a caller does in a [`Lull`]: it gets the engine back for the moment.
pub type LullWork<'a> = dyn FnMut(&mut Engine, Lull) -> Result<(), Box<dyn std::error::Error>> + 'a;

/// `--paired` only: the device copy of each pair's bounds in the packed batch.
/// Seed `j` belongs to pair `p = upper_bound(q_start, j) - 1`, and its hits are
/// the reference positions in `[t_lo[p], t_hi[p])`.
struct PairedBounds {
    q_start: DeviceBuffer<u32>,
    t_lo: DeviceBuffer<u32>,
    t_hi: DeviceBuffer<u32>,
}

/// Hits-per-seed distribution for one chunk.
#[derive(Debug, Default, Clone)]
pub struct HitStats {
    /// Seeds with 0, 1, 2-4, 5-32, 33-256, >256 reference hits.
    pub buckets: [u64; 6],
    /// Hits contributed by each bucket. Seed-weighted buckets alone cannot say
    /// where the *work* is: a heavy-tailed reference puts most seeds in the low
    /// buckets and most hits in the high ones.
    pub hits: [u64; 6],
    pub max: u32,
    pub total_hits: u64,
    pub nonempty: u64,
    /// Lane-slots a warp-per-seed `find_hits` would spend: `32 * ceil(r/32)`
    /// per non-empty seed, since one warp owns one seed. Zero-hit seeds cost
    /// nothing under that mapping.
    warp_slots: u64,
    /// Lane-slots the shipped thread-per-seed mapping spends: `32 * max(r)` per
    /// aligned group of 32 consecutive seeds, because the warp steps together
    /// until its longest walk finishes.
    thread_slots: u64,
    /// Rolling state for `thread_slots`: seeds in the current group of 32 and
    /// the longest walk seen in it.
    group_n: u32,
    group_max: u32,
    /// Hit counts of every non-empty seed, for the median. Only collected when
    /// [`Engine::collect_hit_stats`] is set.
    pub counts: Vec<u32>,
}

impl HitStats {
    pub const LABELS: [&'static str; 6] = ["0", "1", "2-4", "5-32", "33-256", ">256"];

    fn observe(&mut self, n: u32) {
        let b = match n {
            0 => 0,
            1 => 1,
            2..=4 => 2,
            5..=32 => 3,
            33..=256 => 4,
            _ => 5,
        };
        self.buckets[b] += 1;
        self.hits[b] += n as u64;
        self.max = self.max.max(n);
        self.total_hits += n as u64;
        if n > 0 {
            self.nonempty += 1;
            self.counts.push(n);
            self.warp_slots += 32 * n.div_ceil(32) as u64;
        }
        // `observe` is called in `d_hit_num` order, which is the order
        // `find_hits` assigns threads, so 32 consecutive seeds are one warp.
        self.group_max = self.group_max.max(n);
        self.group_n += 1;
        if self.group_n == 32 {
            self.close_group();
        }
    }

    /// Observes one `find_hits` launch. Warp grouping restarts at every launch,
    /// including launches split by `MAX_HITS`.
    fn observe_launch(&mut self, counts: &[u32]) {
        for &n in counts {
            self.observe(n);
        }
        self.close_group();
    }

    /// Charges the in-progress warp group to `thread_slots`. Idempotent.
    fn close_group(&mut self) {
        if self.group_n > 0 {
            self.thread_slots += 32 * self.group_max as u64;
            self.group_n = 0;
            self.group_max = 0;
        }
    }

    /// Total seeds observed, across all `observe` calls.
    pub fn seeds(&self) -> u64 {
        self.buckets.iter().sum()
    }

    /// Folds another engine's distribution into this one, so the multi-bin
    /// executor can report one `--hit-stats` table instead of per-engine ones.
    pub fn merge(&mut self, other: &HitStats) {
        for (a, &b) in self.buckets.iter_mut().zip(&other.buckets) {
            *a += b;
        }
        for (a, &b) in self.hits.iter_mut().zip(&other.hits) {
            *a += b;
        }
        self.max = self.max.max(other.max);
        self.total_hits += other.total_hits;
        self.nonempty += other.nonempty;
        self.counts.extend_from_slice(&other.counts);
        // Engines have independent launch streams, so their trailing groups
        // must be charged separately rather than joined across the merge.
        self.close_group();
        self.warp_slots += other.warp_slots;
        self.thread_slots += other.thread_slots + 32 * other.group_max as u64;
    }

    /// Share of hits living in seeds long enough to fill a warp (`r >= 33`).
    /// This, not mean hits/seed, is what a warp-per-seed mapping can coalesce.
    pub fn hit_share_warp_filling(&self) -> f64 {
        if self.total_hits == 0 {
            return 0.0;
        }
        (self.hits[4] + self.hits[5]) as f64 / self.total_hits as f64 * 100.0
    }

    /// Fraction of issued lane-slots that do useful work under each mapping.
    /// Returns `(thread_per_seed, warp_per_seed)` as percentages.
    pub fn lane_utilisation(&mut self) -> (f64, f64) {
        self.close_group();
        let pct = |slots: u64| {
            if slots == 0 {
                0.0
            } else {
                self.total_hits as f64 / slots as f64 * 100.0
            }
        };
        (pct(self.thread_slots), pct(self.warp_slots))
    }

    /// Mean hits per seed, counting only seeds with at least one hit.
    pub fn mean_nonempty(&self) -> f64 {
        if self.nonempty == 0 {
            0.0
        } else {
            self.total_hits as f64 / self.nonempty as f64
        }
    }

    /// `q`-quantile of the non-empty hit counts, by nearest rank.
    pub fn quantile_nonempty(&mut self, q: f64) -> u32 {
        if self.counts.is_empty() {
            return 0;
        }
        let k = (((self.counts.len() - 1) as f64) * q) as usize;
        *self.counts.select_nth_unstable(k).1
    }

    /// Renders the `--hit-stats` table: the hits-per-seed distribution over
    /// all observed seeds.
    pub fn report(&mut self) -> String {
        let seeds = self.seeds().max(1);
        let hits = self.total_hits.max(1);
        let mut out = String::from("  hits/seed      seeds        %          hits        %\n");
        for (i, (label, n)) in HitStats::LABELS.iter().zip(self.buckets).enumerate() {
            out.push_str(&format!(
                "  {label:<10} {n:>10} {:>7.2}%  {:>12} {:>7.2}%\n",
                n as f64 / seeds as f64 * 100.0,
                self.hits[i],
                self.hits[i] as f64 / hits as f64 * 100.0
            ));
        }
        let mean = self.mean_nonempty();
        let (median, p95, p99) = (
            self.quantile_nonempty(0.5),
            self.quantile_nonempty(0.95),
            self.quantile_nonempty(0.99),
        );
        out.push_str(&format!(
            "  non-empty seeds: mean {mean:.2}  median {median}  p95 {p95}  p99 {p99}  max {}\n",
            self.max
        ));
        out.push_str(&format!(
            "  zero-hit seeds: {} of {} ({:.2}%)\n",
            self.buckets[0],
            self.seeds(),
            self.buckets[0] as f64 / seeds as f64 * 100.0
        ));
        // The round-82 gate: mean hits/seed cannot separate "uniformly dense"
        // from "sparse with a repeat tail", and the two want different
        // `find_hits` mappings.
        let warp_filling = self.hit_share_warp_filling();
        let (thread_util, warp_util) = self.lane_utilisation();
        out.push_str(&format!(
            "  hits in warp-filling seeds (r>=33): {warp_filling:.2}%\n  lane utilisation: \
             thread-per-seed {thread_util:.2}%  warp-per-seed {warp_util:.2}%\n"
        ));
        out
    }
}

/// `find_hsps` behaviour, reduced from the per-hit records the `counters`
/// feature makes the kernel emit.
///
/// Tile counts are histogrammed rather than stored per hit: a mammalian block
/// has 162 M hits, and 1.3 GB of raw samples buys nothing a histogram cannot
/// answer exactly.
/// One candidate's tile-quantized evaluated interval on its diagonal.
#[cfg(feature = "counters")]
#[derive(Debug, Clone, Copy)]
struct Interval {
    diagonal: i32,
    lo: i32,
    hi: i32,
}

/// Counters for the M7/M8 diagonal analysis: evaluated intervals per strand,
/// right/left extension lengths, totals, and the derived per-diagonal view.
#[cfg(feature = "counters")]
#[derive(Debug, Clone)]
pub struct HspStats {
    /// Evaluated intervals per strand, for the M7/M8 diagonal analysis.
    intervals: [Vec<Interval>; 2],
    strand: usize,
    right: Vec<u64>,
    left: Vec<u64>,
    total: Vec<u64>,
    /// Terminations by [x-drop, reference edge, query edge], right then left.
    pub right_term: [u64; 3],
    pub left_term: [u64; 3],
    pub hits: u64,
    pub score_gate_survivors: u64,
    pub entropy_band: u64,
    pub entropy_computed: u64,
    pub accepted: u64,
    pub right_sum: u64,
    pub left_sum: u64,
    pub max_right: u64,
    pub max_left: u64,
    /// First X-drop lane in the first tile of each direction, 0..=32 (63 = the
    /// direction never ran a tile).
    pub drop_r: [u64; 64],
    pub drop_l: [u64; 64],
}

#[cfg(feature = "counters")]
impl Default for HspStats {
    fn default() -> Self {
        // Last bucket is the overflow bin; extensions beyond 4095 tiles are
        // 131k bases and effectively do not happen.
        const N: usize = 4097;
        HspStats {
            intervals: [Vec::new(), Vec::new()],
            strand: 0,
            right: vec![0; N],
            left: vec![0; N],
            total: vec![0; N],
            right_term: [0; 3],
            left_term: [0; 3],
            hits: 0,
            score_gate_survivors: 0,
            entropy_band: 0,
            entropy_computed: 0,
            accepted: 0,
            right_sum: 0,
            left_sum: 0,
            max_right: 0,
            max_left: 0,
            drop_r: [0; 64],
            drop_l: [0; 64],
        }
    }
}

#[cfg(feature = "counters")]
impl HspStats {
    fn bump(hist: &mut [u64], v: u64) {
        hist[(v as usize).min(hist.len() - 1)] += 1;
    }

    /// Which strand subsequent [`observe`](Self::observe) calls belong to.
    pub fn set_strand(&mut self, rev: bool) {
        self.strand = rev as usize;
    }

    #[cfg(feature = "dense-anchors")]
    fn observe_score_gate(&mut self, hits: u32, survivors: u32) {
        self.hits += hits as u64;
        self.score_gate_survivors += survivors as u64;
        let rejected = (hits - survivors) as u64;
        self.right[0] += rejected;
        self.left[0] += rejected;
        self.total[0] += rejected;
    }

    pub fn observe(&mut self, records: &[u64]) {
        for pair in records.chunks_exact(2) {
            let (r, anchor) = (pair[0], pair[1]);
            // A score-gated hit leaves its zero-initialized counter record
            // untouched because the production materializer never ran.
            if r == 0 {
                continue;
            }
            let right = (r & 0xF_FFFF) as i64;
            let left = ((r >> 20) & 0xF_FFFF) as i64;
            let ref_loc = (anchor & 0xFFFF_FFFF) as i64;
            let query_loc = (anchor >> 32) as i64;
            // Evaluated interval on the reference, tile-quantized and including
            // the terminating tile: every lane loads its cell before the
            // first-drop ballot fires, so the read extends past the final max.
            self.intervals[self.strand].push(Interval {
                diagonal: (ref_loc - query_loc) as i32,
                lo: (ref_loc - left * 32) as i32,
                hi: (ref_loc + right * 32 - 1) as i32,
            });
        }
        for &r in records.iter().step_by(2) {
            let right = r & 0xF_FFFF;
            let left = (r >> 20) & 0xF_FFFF;
            #[cfg(not(feature = "dense-anchors"))]
            {
                self.hits += 1;
            }
            if r == 0 {
                Self::bump(&mut self.right, 0);
                Self::bump(&mut self.left, 0);
                Self::bump(&mut self.total, 0);
                continue;
            }
            #[cfg(not(feature = "dense-anchors"))]
            {
                self.score_gate_survivors += 1;
            }
            self.right_sum += right;
            self.left_sum += left;
            self.max_right = self.max_right.max(right);
            self.max_left = self.max_left.max(left);
            Self::bump(&mut self.right, right);
            Self::bump(&mut self.left, left);
            Self::bump(&mut self.total, right + left);
            for (shift, term) in [(40, 0), (43, 1)] {
                let code = (r >> shift) & 0b111;
                let slot = match code {
                    1 => 0, // x-drop
                    2 => 1, // reference boundary
                    4 => 2, // query boundary
                    _ => continue,
                };
                if term == 0 {
                    self.right_term[slot] += 1
                } else {
                    self.left_term[slot] += 1
                }
            }
            self.drop_r[((r >> 49) & 63) as usize] += 1;
            self.drop_l[((r >> 55) & 63) as usize] += 1;
            self.entropy_band += (r >> 46) & 1;
            self.entropy_computed += (r >> 47) & 1;
            self.accepted += (r >> 48) & 1;
        }
    }

    fn quantile(hist: &[u64], n: u64, q: f64) -> usize {
        let target = (n as f64 * q) as u64;
        let mut seen = 0u64;
        for (v, &c) in hist.iter().enumerate() {
            seen += c;
            if seen >= target {
                return v;
            }
        }
        hist.len() - 1
    }

    /// Share of all tile work done by the busiest `frac` of hits.
    fn top_work_share(hist: &[u64], frac: f64) -> f64 {
        let n: u64 = hist.iter().sum();
        let total: u64 = hist.iter().enumerate().map(|(v, &c)| v as u64 * c).sum();
        let budget = (n as f64 * frac) as u64;
        let (mut taken, mut work) = (0u64, 0u64);
        for (v, &c) in hist.iter().enumerate().rev() {
            let take = c.min(budget.saturating_sub(taken));
            work += take * v as u64;
            taken += take;
            if taken >= budget {
                break;
            }
        }
        if total == 0 {
            0.0
        } else {
            work as f64 / total as f64 * 100.0
        }
    }

    /// Diagonal structure (M7) and repeated-cell accounting (M8).
    ///
    /// Intervals are sorted by (diagonal, start) once; each diagonal group then
    /// gets a coverage sweep, which yields the union length and the depth
    /// profile in the same pass.
    pub fn diagonal_report(&mut self, find_hsps_share: f64) -> String {
        let mut out = String::new();
        let (mut tot_cells, mut uniq_cells) = (0u64, 0u64);
        let (mut deep2, mut deep4, mut deep8) = (0u64, 0u64, 0u64);
        let (mut work2, mut work4, mut work8) = (0u64, 0u64, 0u64);

        for (strand, label) in [(0usize, "plus"), (1usize, "minus")] {
            let iv = &mut self.intervals[strand];
            if iv.is_empty() {
                continue;
            }
            iv.sort_unstable_by_key(|i| (i.diagonal, i.lo));

            let mut per_diag: Vec<u64> = Vec::new();
            let mut gaps: Vec<i64> = Vec::new();
            let mut ends: Vec<i32> = Vec::new();
            let mut i = 0usize;
            while i < iv.len() {
                let d = iv[i].diagonal;
                let mut j = i;
                while j < iv.len() && iv[j].diagonal == d {
                    j += 1;
                }
                per_diag.push((j - i) as u64);
                for k in i + 1..j {
                    gaps.push((iv[k].lo - iv[k - 1].lo) as i64);
                }

                // Coverage sweep over this diagonal: starts are already sorted,
                // so sort the ends and walk both.
                ends.clear();
                ends.extend(iv[i..j].iter().map(|x| x.hi + 1));
                ends.sort_unstable();
                let (mut si, mut ei, mut depth, mut pos) = (i, 0usize, 0i64, iv[i].lo as i64);
                while ei < ends.len() {
                    let next = if si < j {
                        (iv[si].lo as i64).min(ends[ei] as i64)
                    } else {
                        ends[ei] as i64
                    };
                    if next > pos && depth > 0 {
                        let len = (next - pos) as u64;
                        uniq_cells += len;
                        tot_cells += len * depth as u64;
                        if depth >= 2 {
                            deep2 += len;
                            work2 += len * depth as u64;
                        }
                        if depth >= 4 {
                            deep4 += len;
                            work4 += len * depth as u64;
                        }
                        if depth >= 8 {
                            deep8 += len;
                            work8 += len * depth as u64;
                        }
                    }
                    pos = next;
                    while si < j && iv[si].lo as i64 == pos {
                        depth += 1;
                        si += 1;
                    }
                    while ei < ends.len() && ends[ei] as i64 == pos {
                        depth -= 1;
                        ei += 1;
                    }
                }
                i = j;
            }

            let q = |v: &mut Vec<u64>, p: f64| -> u64 {
                if v.is_empty() {
                    return 0;
                }
                let k = (((v.len() - 1) as f64) * p) as usize;
                *v.select_nth_unstable(k).1
            };
            let diags = per_diag.len() as u64;
            let hits: u64 = per_diag.iter().sum();
            out.push_str(&format!(
                "  {label:<5} diagonals {diags:>12}  hits {hits:>13}  hits/diag mean {:.2}                  median {} p95 {} p99 {} max {}\n",
                hits as f64 / diags.max(1) as f64,
                q(&mut per_diag, 0.5),
                q(&mut per_diag, 0.95),
                q(&mut per_diag, 0.99),
                per_diag.iter().copied().max().unwrap_or(0)
            ));
            if !gaps.is_empty() {
                let mut g: Vec<u64> = gaps.iter().map(|&x| x as u64).collect();
                out.push_str(&format!(
                    "  {label:<5} same-diagonal neighbour distance  p10 {}  p50 {}  p95 {}  p99 {}\n",
                    q(&mut g, 0.10),
                    q(&mut g, 0.5),
                    q(&mut g, 0.95),
                    q(&mut g, 0.99)
                ));
            }
        }

        let redundancy = tot_cells as f64 / uniq_cells.max(1) as f64;
        let kernel_ceiling = 1.0 - 1.0 / redundancy;
        out.push_str(&format!(
            "  -- M8 repeated-cell accounting --\n  total evaluated cells {tot_cells}\n               unique diagonal cells {uniq_cells}\n  redundancy factor {redundancy:.3}\n"
        ));
        out.push_str(&format!(
            "  unique cells covered >=2x {:.2}%  >=4x {:.2}%  >=8x {:.2}%\n",
            deep2 as f64 / uniq_cells.max(1) as f64 * 100.0,
            deep4 as f64 / uniq_cells.max(1) as f64 * 100.0,
            deep8 as f64 / uniq_cells.max(1) as f64 * 100.0
        ));
        out.push_str(&format!(
            "  evaluated work on those regions >=2x {:.2}%  >=4x {:.2}%  >=8x {:.2}%\n",
            work2 as f64 / tot_cells.max(1) as f64 * 100.0,
            work4 as f64 / tot_cells.max(1) as f64 * 100.0,
            work8 as f64 / tot_cells.max(1) as f64 * 100.0
        ));
        out.push_str(&format!(
            "  ceiling: find_hsps {:.1}%  whole runtime {:.1}% (find_hsps share {:.0}%)\n",
            kernel_ceiling * 100.0,
            kernel_ceiling * find_hsps_share * 100.0,
            find_hsps_share * 100.0
        ));
        out
    }

    /// Renders the `counters` feature's find_hsps statistics table.
    pub fn report(&self) -> String {
        let n = self.hits.max(1);
        let mut o = String::new();
        let line = |o: &mut String, name: &str, hist: &[u64], sum: u64, max: u64| {
            o.push_str(&format!(
                "  {name:<6} mean {:.2}  median {}  p95 {}  p99 {}  p99.9 {}  max {}\n",
                sum as f64 / n as f64,
                Self::quantile(hist, n, 0.5),
                Self::quantile(hist, n, 0.95),
                Self::quantile(hist, n, 0.99),
                Self::quantile(hist, n, 0.999),
                max
            ));
        };
        o.push_str("  -- 2.1 extension length, in 32-base tiles --\n");
        line(&mut o, "right", &self.right, self.right_sum, self.max_right);
        line(&mut o, "left", &self.left, self.left_sum, self.max_left);
        line(
            &mut o,
            "both",
            &self.total,
            self.right_sum + self.left_sum,
            self.max_right + self.max_left,
        );
        o.push_str(&format!(
            "  tiles total {} (~{} Mbase scanned)\n",
            self.right_sum + self.left_sum,
            (self.right_sum + self.left_sum) * 32 / 1_000_000
        ));

        o.push_str("  -- 2.2 termination --\n");
        for (name, t) in [("right", self.right_term), ("left", self.left_term)] {
            let s = t.iter().sum::<u64>().max(1);
            o.push_str(&format!(
                "  {name:<6} x-drop {:>12} ({:5.2}%)  ref edge {:>10} ({:5.2}%)  query edge {:>10} ({:5.2}%)\n",
                t[0], t[0] as f64 / s as f64 * 100.0,
                t[1], t[1] as f64 / s as f64 * 100.0,
                t[2], t[2] as f64 / s as f64 * 100.0
            ));
        }

        o.push_str("  -- 2.3 candidate funnel --\n");
        o.push_str(&format!(
            "  {:<28} {:>12}  {:8.4}%\n",
            "seed hits into find_hsps", self.hits, 100.0
        ));
        if cfg!(feature = "warp-score-gate") {
            let v = self.score_gate_survivors;
            o.push_str(&format!(
                "  {:<28} {v:>12}  {:8.4}%\n",
                "score-gate survivors",
                v as f64 / n as f64 * 100.0
            ));
        }
        for (name, v) in [
            ("in entropy score band", self.entropy_band),
            ("entropy actually computed", self.entropy_computed),
            ("accepted (raw HSPs)", self.accepted),
        ] {
            o.push_str(&format!(
                "  {name:<28} {v:>12}  {:8.4}%\n",
                v as f64 / n as f64 * 100.0
            ));
        }

        o.push_str("  -- 2.4 work distribution (both directions) --\n");
        let buckets: [(&str, usize, usize); 6] = [
            ("1", 1, 1),
            ("2-4", 2, 4),
            ("5-16", 5, 16),
            ("17-64", 17, 64),
            ("65-256", 65, 256),
            (">256", 257, usize::MAX),
        ];
        o.push_str(&format!(
            "  {:<8} {:>12} {:>8} {:>14} {:>8}\n",
            "tiles", "hits", "%", "tile work", "% work"
        ));
        let total_work: u64 = self
            .total
            .iter()
            .enumerate()
            .map(|(v, &c)| v as u64 * c)
            .sum();
        let zero = self.total[0];
        o.push_str(&format!(
            "  {:<8} {zero:>12} {:>7.2}% {:>14} {:>7.2}%\n",
            "0",
            zero as f64 / n as f64 * 100.0,
            0,
            0.0
        ));
        for (label, lo, hi) in buckets {
            let hits: u64 = self
                .total
                .iter()
                .enumerate()
                .filter(|(v, _)| *v >= lo && *v <= hi)
                .map(|(_, &c)| c)
                .sum();
            let work: u64 = self
                .total
                .iter()
                .enumerate()
                .filter(|(v, _)| *v >= lo && *v <= hi)
                .map(|(v, &c)| v as u64 * c)
                .sum();
            o.push_str(&format!(
                "  {label:<8} {hits:>12} {:>7.2}% {work:>14} {:>7.2}%\n",
                hits as f64 / n as f64 * 100.0,
                work as f64 / total_work.max(1) as f64 * 100.0
            ));
        }
        o.push_str(&format!(
            "  top 1% of hits do {:.2}% of tile work; top 0.1% do {:.2}%\n",
            Self::top_work_share(&self.total, 0.01),
            Self::top_work_share(&self.total, 0.001)
        ));

        // -- M11: exact short-extension distribution and per-direction shape --
        o.push_str("  -- M11 exact tile distribution (right + left) --\n");
        let work = |h: &[u64], lo: usize, hi: usize| -> u64 {
            h.iter()
                .enumerate()
                .filter(|(v, _)| *v >= lo && *v <= hi)
                .map(|(v, &c)| v as u64 * c)
                .sum()
        };
        let hits = |h: &[u64], lo: usize, hi: usize| -> u64 {
            h.iter()
                .enumerate()
                .filter(|(v, _)| *v >= lo && *v <= hi)
                .map(|(_, &c)| c)
                .sum()
        };
        let total_work = work(&self.total, 0, usize::MAX);
        o.push_str(&format!(
            "  {:<8} {:>12} {:>8} {:>14} {:>8}\n",
            "tiles", "hits", "%", "tile work", "% work"
        ));
        for (label, lo, hi) in [
            ("<=1", 0usize, 1usize),
            ("2", 2, 2),
            ("3", 3, 3),
            ("4", 4, 4),
            ("5-8", 5, 8),
            (">8", 9, usize::MAX),
        ] {
            let (hc, wc) = (hits(&self.total, lo, hi), work(&self.total, lo, hi));
            o.push_str(&format!(
                "  {label:<8} {hc:>12} {:>7.2}% {wc:>14} {:>7.2}%\n",
                hc as f64 / n as f64 * 100.0,
                wc as f64 / total_work.max(1) as f64 * 100.0
            ));
        }
        o.push_str("  -- first X-drop lane, first tile of each direction --\n");
        for (label, h) in [("right", &self.drop_r), ("left", &self.drop_l)] {
            let seen: u64 = h.iter().take(33).sum();
            let cum = |k: usize| -> f64 {
                h.iter().take(k + 1).sum::<u64>() as f64 / seen.max(1) as f64 * 100.0
            };
            let buckets = [
                ("0-7", 0usize, 7usize),
                ("8-15", 8, 15),
                ("16-23", 16, 23),
                ("24-31", 24, 31),
            ];
            let mut line = format!("  {label:<6}");
            for (n, lo, hi) in buckets {
                let c: u64 = h.iter().take(hi + 1).skip(lo).sum();
                line.push_str(&format!(
                    "  {n} {:5.2}%",
                    c as f64 / seen.max(1) as f64 * 100.0
                ));
            }
            line.push_str(&format!(
                "  no-drop {:5.2}%",
                h[32] as f64 / seen.max(1) as f64 * 100.0
            ));
            o.push_str(&line);
            o.push_str(&format!(
                "\n         cumulative: <=8 {:5.2}%  <=16 {:5.2}%  <=24 {:5.2}%\n",
                cum(8),
                cum(16),
                cum(24)
            ));
        }
        o.push_str("  -- M11 per-direction shape --\n");
        o.push_str(&format!(
            "  terminate in the FIRST right tile   {:>12} {:>7.2}%\n",
            self.right[1],
            self.right[1] as f64 / n as f64 * 100.0
        ));
        o.push_str(&format!(
            "  never enter the left loop           {:>12} {:>7.2}%\n",
            self.left[0],
            self.left[0] as f64 / n as f64 * 100.0
        ));
        o.push_str(&format!(
            "  left loop ends in its first tile    {:>12} {:>7.2}%\n",
            self.left[1],
            self.left[1] as f64 / n as f64 * 100.0
        ));
        o.push_str(&format!(
            "  right tiles {} ({:.1}%) vs left tiles {} ({:.1}%)\n",
            self.right_sum,
            self.right_sum as f64 / (self.right_sum + self.left_sum).max(1) as f64 * 100.0,
            self.left_sum,
            self.left_sum as f64 / (self.right_sum + self.left_sum).max(1) as f64 * 100.0
        ));
        o.push_str("  LEFT loop shape (it owns ~62% of tile work and every candidate enters it)\n");
        for k in 1..=3usize {
            o.push_str(&format!(
                "    left = {k} tile(s) {:>12} {:>7.2}%\n",
                self.left[k],
                self.left[k] as f64 / n as f64 * 100.0
            ));
        }
        let left_deep: u64 = self.left.iter().skip(4).sum();
        o.push_str(&format!(
            "    left >= 4 tiles  {:>12} {:>7.2}%\n",
            left_deep,
            left_deep as f64 / n as f64 * 100.0
        ));
        o.push_str(&format!(
            "  right<=1 AND left<=1 (a 2-tile fast path would cover) {:>10} {:>7.2}%\n",
            hits(&self.right, 0, 1).min(hits(&self.left, 0, 1)),
            hits(&self.total, 0, 2) as f64 / n as f64 * 100.0
        ));
        o
    }
}

/// Everything the GPU stage needs that outlives a single chunk.
///
/// Scoped to one *reference bin*: construction uploads the
/// reference index, reference sequence and scoring matrix, and every query bin
/// then arrives through [`swap_query`](Engine::swap_query). One `Engine` per
/// reference bin, not per work unit — the difference is ~1 GB of `pos_table`
/// re-upload per avoided construction.
pub struct Engine {
    stream: Arc<CudaStream>,
    module: kernels::device::LoadedModule,
    index_table: DeviceBuffer<u32>,
    pos_table: DeviceBuffer<u32>,
    ref_seq: DeviceBuffer<u8>,
    query_seq: DeviceBuffer<u8>,
    query_rc_seq: DeviceBuffer<u8>,
    sub_mat: DeviceBuffer<i32>,
    ref_len: u32,
    query_len: u32,
    seed_size: u32,
    xdrop: i32,
    hspthresh: i32,
    /// What the score-gate kernels receive as `hspthresh`: `hspthresh` itself,
    /// or with `gate-int-keep` its [`keep_threshold`]. Resolved once here.
    #[cfg_attr(not(feature = "dense-anchors"), allow(dead_code))]
    gate_thresh: i32,
    /// `gate-sieve`: the packed planes and this unit's class scores for the
    /// lane-local bound in front of the gate (`gpu/sieve.rs`).
    #[cfg(feature = "gate-sieve")]
    sieve: GateSieve,
    noentropy: u32,
    max_hits: u32,
    hit_capacity: u32,
    timing: bool,
    /// N8: `find_hsps` grid, resolved once from the config.
    hsp_blocks: u32,
    /// Keep the pre-dedup anchors in [`FilterOutput::raw`].
    pub dump_raw: bool,
    /// `--paired` bounds (see [`set_pairs`](Self::set_pairs)). `None` on every
    /// other path, which therefore launches exactly the kernels it always did.
    paired: Option<PairedBounds>,
    /// Round 122 (the transition fold): the table set that is not in use, as
    /// index, positions, bucket counts and bucket rows. `fold_active` says
    /// whether the engine's own four are the folded set (every batch, unless
    /// the caller sent plain seeds) or the plain one.
    fold_alt: Option<Box<[DeviceBuffer<u32>; 4]>>,
    fold_active: bool,
    /// The next batch call is on plain seeds ([`plain_next`](Self::plain_next)).
    plain_once: bool,
    /// The largest chunk the card holds beside the folded tables.
    fold_capacity: u32,
    /// Folded batches this engine was given, and how many of them came back
    /// to be sent again.
    fold_seen: u32,
    fold_again: u32,
    /// `--paired`: set by the driver before each call — whether this call's
    /// seeds span more than one pair, the only case that may stop as
    /// [`FilterOutput::oversized`].
    pub paired_multi: bool,
    /// `--paired`: each seed's first in-pair position within its bucket.
    buf_seed_lo: DeviceBuffer<u32>,
    /// Accumulate the hits-per-seed distribution.
    pub collect_hit_stats: bool,
    /// S0 survivor audit (`HSPZ_ANCHOR_CENSUS`). Off the timed path; folds into
    /// scalars and never accumulates anchors.
    pub census: Option<crate::census::SurvivorAudit>,
    pub phases: Phases,
    pub hit_stats: HitStats,
    #[cfg(feature = "counters")]
    pub hsp_stats: HspStats,
    #[cfg(feature = "counters")]
    pub groups: crate::hsp::Groups,
    /// Round 85: `HSPZ_CHUNK_WALK=full` forces the pre-round-85 full
    /// cumulative-array copy in the over-cap chunk walk; default (unset or any
    /// other value) is the sparse block walk. Read once here, not per call.
    chunk_walk_full: bool,
    /// Round 85: reused DMA target for the sparse walk's <=256-element block
    /// fetches, so the hot path reuses one small host buffer instead of
    /// allocating a fresh pageable Vec per fetch.
    chunk_block_scratch: Vec<u32>,
    /// Kernel launches issued so far, for pricing launch overhead.
    pub launches: u64,
    /// High-water device usage, sampled at the point of peak allocation.
    pub peak_used: usize,
    /// Persistent per-batch work buffers. The control sizes HSP/status by raw
    /// hits; the dense path sizes anchors/flags by raw hits and HSP/status only
    /// by score-gate survivors. Spare capacity is never read.
    buf_hsp: DeviceBuffer<SegmentPair>,
    buf_done: DeviceBuffer<u32>,
    #[cfg(feature = "dense-anchors")]
    buf_anchor: DeviceBuffer<u64>,
    #[cfg(feature = "dense-anchors")]
    buf_flags: DeviceBuffer<u8>,
    #[cfg(feature = "dense-anchors")]
    buf_survivors: DeviceBuffer<u32>,
    /// Round 83 (`ref-loc-buckets`): the chunk's anchors permuted into
    /// reference-address buckets, the raw hit id each permuted slot came from,
    /// and the bucket-major per-block histogram the permutation is built from.
    #[cfg(feature = "ref-loc-buckets")]
    buf_sorted_anchor: DeviceBuffer<u64>,
    #[cfg(feature = "ref-loc-buckets")]
    buf_sorted_idx: DeviceBuffer<u32>,
    #[cfg(feature = "ref-loc-buckets")]
    buf_bucket_counts: DeviceBuffer<u32>,
    /// Round 2: one `u32` of gate keep bits per 32 sorted slots, in place of the
    /// byte flags. Sized `ceil(n/32)` and fully overwritten every chunk.
    #[cfg(feature = "ref-loc-buckets")]
    buf_flags_bits: DeviceBuffer<u32>,
    /// Window size (`1 << bucket_shift` bytes) and bucket count, fixed for the
    /// engine's reference. `n_buckets <= 32` is a hard invariant of the ballot
    /// loop in `bucket_count`/`bucket_scatter`.
    /// The bucket window is sized for the sieve's packed planes (a quarter of
    /// a byte per base), not for the byte reference: a chunk the sieve does not
    /// gate (the unit switched it off, or its guard holds the plain gate) then
    /// skips the bucket pass, whose window would not fit the L2 as bytes.
    #[cfg(feature = "ref-loc-buckets")]
    bucket_planes: bool,
    /// Round 119 (`HSPZ_HITS_BUCKETED`, see `HitsBucketed`): a chunk's bucket
    /// order written from its seeds. `buf_bucket_bounds` holds a row of bucket
    /// edges for each k-mer whose counts do not fit `buf_bucket_nibbles`, and
    /// `buf_seed_counts` a chunk's hits per bucket and run of 32 seeds.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    hits_bucketed: HitsBucketed,
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    buf_bucket_bounds: DeviceBuffer<u32>,
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    buf_seed_counts: DeviceBuffer<u32>,
    /// Every k-mer's hits per bucket in four bits each (`bucket_nibbles`; empty
    /// where the path does not apply), and the words of a chunk's seeds in seed
    /// order.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    buf_bucket_nibbles: DeviceBuffer<u32>,
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    buf_seed_nib: DeviceBuffer<u32>,
    /// Round 116: a bucketed chunk's survivors go to `find_hsps` by sorted slot
    /// (`HSPZ_SURVIVOR_ORDER=raw` restores raw hit order).
    #[cfg(all(feature = "ref-loc-buckets", feature = "gate-sieve"))]
    survivor_slots: bool,
    /// Round 114: the bucket kernels learn which lanes of a warp share a bucket
    /// from one `match.any` (`true`, the default) or from a vote per bucket
    /// (`HSPZ_BUCKET_VOTES=loop`).
    #[cfg(feature = "ref-loc-buckets")]
    bucket_match: bool,
    /// Round 110 (`find-hits-groups`, experiment): which kernel expands the
    /// seeds of a chunk into anchors.
    #[cfg(feature = "find-hits-groups")]
    find_hits_groups: FindHitsGroups,
    #[cfg(feature = "ref-loc-buckets")]
    bucket_shift: u32,
    #[cfg(feature = "ref-loc-buckets")]
    n_buckets: u32,
    /// Cycle 4 (round 87, blocks): the runtime auto/off/on policy for the
    /// pass. `HSPZ_REF_BUCKETS=0`/`=1` force it off/on; `=auto` runs
    /// alternating LONG blocks of each production path (OFF, ON x3) and
    /// commits to the faster settled ns/hit; geometry auto-off (a window
    /// that does not fit 3/4 of L2) forces it off before any block. A pinned
    /// `HSPZ_REF_BUCKET_SHIFT` always reports "fits", which is how it forces
    /// the pass on under `auto` too.
    #[cfg(feature = "ref-loc-buckets")]
    bucket_mode: BucketMode,
    /// Round 87 (blocks): the gate+count span event pair for the chunk's
    /// production path, created lazily on the first timed chunk and
    /// re-recorded every timed chunk (never a sync: the elapsed value is
    /// read only after the producer's count sync, which stream order
    /// guarantees covers the span).
    #[cfg(feature = "ref-loc-buckets")]
    block_span_start: Option<CudaEvent>,
    #[cfg(feature = "ref-loc-buckets")]
    block_span_end: Option<CudaEvent>,
    /// Round 87: ledger identity. Engine index (process-wide), so each
    /// block line names its engine.
    #[cfg(feature = "ref-loc-buckets")]
    bucket_eng_id: u32,
    /// N3: reuse `d_seeds` / `d_hit_num` across batches instead of
    /// allocating and freeing them per batch. Off by default; the earlier
    /// `d_hsp`/`d_done` result showed `cuMemFree` can cost more than the named
    /// allocation stage, so this is judged on whole runtime, not on alloc time.
    pub persistent_seed_buffers: bool,
    /// Phase 3: two seed buffers, so batch *N+1*'s upload can be in flight while
    /// batch *N*'s kernels read the other one. Slot parity is `batch % 2`.
    seed_slots: [DeviceBuffer<u64>; 2],
    seed_len: [u32; 2],
    buf_hit_num: DeviceBuffer<u32>,
    seed_shape: DeviceBuffer<u32>,
    buf_seed_kmer: DeviceBuffer<u32>,
    /// Retained until `seed_and_filter` drains the stream; a per-call temporary
    /// could be freed while the async add-offset kernel still reads it.
    buf_seed_offsets: DeviceBuffer<u32>,
    /// Round 121: the small buffers every batch needs for its scans and counts
    /// (block sums and the offsets the host sends back), the materializer's
    /// unused statistics word and the reduced output. Each was allocated and
    /// freed once per batch: nine `cuMemAlloc`/`cuMemFree` pairs and three
    /// memsets between kernels that wait on nothing else. They are grown like
    /// every other buffer ([`reserve`]) and every element a kernel reads is
    /// written first, so none needs zeroing. A use is always separated from the
    /// next by a wait on the stream (each scan's sums are read by the host).
    scr_hit_sums: DeviceBuffer<u32>,
    scr_hit_offsets: DeviceBuffer<u32>,
    #[cfg(feature = "ref-loc-buckets")]
    scr_run_sums: DeviceBuffer<u32>,
    #[cfg(feature = "dense-anchors")]
    scr_surv_sums: DeviceBuffer<u32>,
    #[cfg(feature = "dense-anchors")]
    scr_surv_offsets: DeviceBuffer<u32>,
    scr_done_sums: DeviceBuffer<u32>,
    scr_done_offsets: DeviceBuffer<u32>,
    #[cfg(not(feature = "counters"))]
    scr_stats: DeviceBuffer<u64>,
    buf_reduced: DeviceBuffer<SegmentPair>,
    /// The allocation count ([`ALLOC_EPOCH`]) at the last device-memory sample:
    /// the peak can only have moved if something was allocated since.
    mem_epoch_seen: u64,
    /// Free list of pinned host staging buffers. Page-locking is
    /// expensive — `cuMemHostAlloc` of two 26 MB buffers cost ~13 ms per pass
    /// when this was done per pass — so they live for the engine's lifetime and
    /// are handed out and returned instead of reallocated.
    pinned_slots: Vec<cuda_core::PinnedHostBuffer<u64>>,
    /// Set once a pinned buffer has actually been allocated and returned, so a
    /// silent fallback to pageable memory cannot masquerade as a pinned run.
    pinned_ok: bool,
    /// Round 71: the GPU-timeline gap between one stage's end event and the next
    /// stage's start event, summed. The phase table cannot recover this — it retains
    /// durations, not a shared timeline, and a blocking copy absorbs all queued work
    /// into its own row. `last_end` is deliberately carried *across* `resolve_pending`
    /// calls, because the interesting bubble is exactly at a sync point: the host reads
    /// a scan total back, then enqueues the next kernel.
    last_end: Option<CudaEvent>,
    last_name: &'static str,
    gap_ms: f32,
    gap_n: u64,
    gap_max: f32,
    gap_max_pair: (&'static str, &'static str),
    /// Round 72: per-pair totals. One dominant pair means a specific round trip worth
    /// removing; a flat spread over hundreds of pairs means launch and enqueue overhead,
    /// which is structural. The single largest gap cannot tell those apart. Linear scan
    /// over a handful of distinct stage pairs.
    gap_pairs: Vec<((&'static str, &'static str), f32, u64)>,
    /// Intervals the gap accounting discards because they exceed 1,000 ms
    /// (unit-transition work: query pack, ref prep). Counted separately so
    /// the residual is visible instead of silently dropped.
    gap_long_n: u64,
    gap_long_ms: f32,
    /// Generate the query seed stream on the device instead of walking it on the
    /// host (round 68). Both paths are compiled; the executor sets this from the
    /// worker count because the trade reverses with GPU count — on one GPU the
    /// device seeder adds ~165 s of device work against a host tail that pinned
    /// async H->D already hides (r39/r51), and on two T4s it is worth -20.3% of
    /// wall because that tail is 397 s and exposed.
    pub device_seeds: bool,
    /// Device uploads of the reference index. One per `Engine`
    /// construction; the executor checks it against the reference-bin count.
    reference_uploads: u32,
    /// Query bins run against the resident reference bin.
    query_swaps: u32,
    /// Stages whose CUDA events are recorded but not yet read.
    ///
    /// Measuring a stage must not force it to complete, so `end_stage` records
    /// the end event and moves on; the durations are read at the next genuine
    /// host dependency, when the events are known to have completed.
    pending: Vec<PendingStage>,
    /// The synchronization mechanism gate. `stage_syncs` are the host waits that exist
    /// only to measure or to serialise stages that stream order already
    /// serialises — the ones this phase removes. `pipeline_syncs` are the real
    /// dependencies: a host read of a device result, or a free that must not race
    /// a queued kernel.
    stage_syncs: u64,
    pipeline_syncs: u64,
    /// Drop the per-stage host synchronization (`--async-stages`). Off by
    /// default: round 29 measured the removal as performance-neutral on an L4, so
    /// the shipped path keeps the simpler invariant. Phase 3 needs this path, and
    /// `parity.sh` keeps it honest.
    pub async_stages: bool,
    /// Phase 3: upload seeds on a second stream instead of blocking the host.
    ///
    /// `copy_ready[s]` is recorded on the copy stream when slot *s*'s upload
    /// finishes; the compute stream waits on it before `find_num_hits`.
    /// `compute_done[s]` is recorded on the compute stream when the batch that read
    /// slot *s* is finished; the copy stream waits on it before overwriting. The
    /// pair carries the ordering that today's per-call boundary drain also happens
    /// to give, so relaxing that drain cannot introduce a race here.
    copy_stream: Arc<CudaStream>,
    copy_ready: [CudaEvent; 2],
    compute_done: [CudaEvent; 2],
    /// Timing-enabled start events, created lazily and only under `--time`.
    copy_start: [Option<CudaEvent>; 2],
    pub async_seed_copy: bool,
    /// Mechanism: uploads issued, and how many were still in flight when their
    /// compute needed them. A stall is overlap that did not happen.
    seed_uploads: u64,
    seed_copy_stalls: u64,
}

/// One stage's recorded events, awaiting a synchronization that covers them.
struct PendingStage {
    name: &'static str,
    host: Duration,
    events: Option<(CudaEvent, CudaEvent)>,
}

/// Grows `buf` to at least `len` elements, keeping the existing allocation when
/// it is already big enough. Never shrinks: a pass allocates as many times as
/// the batch high-water mark rises, which in practice is once.
///
/// # Safety
///
/// Same contract as [`uninitialized`] — the producing kernel must write every
/// element the consumer reads. Contents from the previous batch are retained
/// and must not be relied on.
unsafe fn reserve<T>(
    buf: &mut DeviceBuffer<T>,
    stream: &CudaStream,
    len: usize,
    syncs: &mut u64,
) -> Result<(), DriverError> {
    if buf.len() < len {
        // Freeing memory a queued kernel still references is UB,
        // and with the per-stage syncs gone the stream is no longer conveniently
        // drained by the time we get here. Wait once, on the grow path only — it
        // fires when the batch high-water mark rises, in practice once per pass,
        // and it shows up in `pipeline_syncs` rather than hiding inside cuMemFree.
        stream.synchronize()?;
        *syncs += 1;
        // Drop the old allocation before taking the new one so peak device
        // usage stays at one buffer, not two.
        *buf = unsafe { uninitialized::<T>(stream, 0)? };
        *buf = unsafe { uninitialized::<T>(stream, len)? };
    }
    Ok(())
}

/// Copies `data` into `buf` on `stream`, growing `buf` first, and does not wait
/// for the copy: the kernels queued after it on the same stream see it.
///
/// # Safety
///
/// As [`reserve`]; and `data` must stay alive until the stream is next
/// synchronized (the driver stages a pageable source before it returns, so
/// this is belt and braces).
unsafe fn upload<T: DeviceCopy>(
    buf: &mut DeviceBuffer<T>,
    stream: &CudaStream,
    data: &[T],
    syncs: &mut u64,
) -> Result<(), DriverError> {
    unsafe { reserve(buf, stream, data.len(), syncs)? };
    if data.is_empty() {
        return Ok(());
    }
    // SAFETY: `buf` holds at least `data.len()` elements after the reserve.
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            buf.cu_deviceptr(),
            data.as_ptr(),
            std::mem::size_of_val(data),
            stream.cu_stream(),
        )
    }
}

/// `HSPZ_HOST_TRACE=1`: where the host's time inside the batch calls goes,
/// without `--time` (whose events add host work of their own): blocked on the
/// card (the card is busy and the host waits), doing work under queued kernels
/// (a [`Lull`]), or doing its own work with nothing queued, which is the time
/// the card may sit idle. Summed over all workers; printed at the end of a run.
pub mod host_trace {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static BATCH_NS: AtomicU64 = AtomicU64::new(0);
    pub static BLOCKED_NS: AtomicU64 = AtomicU64::new(0);
    pub static LULL_NS: AtomicU64 = AtomicU64::new(0);
    pub static BATCHES: AtomicU64 = AtomicU64::new(0);

    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("HSPZ_HOST_TRACE").as_deref() == Ok("1"))
    }

    /// Runs `wait` and, when tracing, charges its duration to `to`.
    pub fn timed<T>(to: &AtomicU64, wait: impl FnOnce() -> T) -> T {
        if !on() {
            return wait();
        }
        let t = std::time::Instant::now();
        let out = wait();
        to.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        out
    }
}

/// Counts device allocations made through [`uninitialized`], so that the
/// device-memory sample of a batch is taken only when one happened.
static ALLOC_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `cuDeviceGetAttribute` for `ordinal`, no context required (round 90b: the
/// unit-partition guard runs before any worker owns a device).
fn device_attribute(ordinal: i32, attr: cuda_core::sys::CUdevice_attribute) -> Result<i32, String> {
    let mut value: i32 = 0;
    // SAFETY: driver query; `value` is a local and the status is checked.
    unsafe {
        if cuda_core::sys::cuDeviceGetAttribute(&mut value, attr, ordinal)
            != cuda_core::sys::cudaError_enum_CUDA_SUCCESS
        {
            return Err(format!("cuDeviceGetAttribute failed for device {ordinal}"));
        }
    }
    Ok(value)
}

/// The three attributes the round-90b same-device guard compares: resident
/// multiprocessors, the driver-reported clock in kHz, and L2 cache bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceProfile {
    pub(crate) sms: i32,
    pub(crate) clock_khz: i32,
    pub(crate) l2_bytes: u64,
}

/// [`DeviceProfile`] of `ordinal`, queried once per used device before any
/// worker spawns (the auto unit-partition guard).
pub(crate) fn device_profile(ordinal: i32) -> Result<DeviceProfile, String> {
    use cuda_core::sys::{
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE as CLOCK_RATE,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE as L2_CACHE_SIZE,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as MULTIPROCESSOR_COUNT,
    };
    Ok(DeviceProfile {
        sms: device_attribute(ordinal, MULTIPROCESSOR_COUNT)?,
        clock_khz: device_attribute(ordinal, CLOCK_RATE)?,
        l2_bytes: device_attribute(ordinal, L2_CACHE_SIZE)?.max(0) as u64,
    })
}

/// Device L2 cache bytes, or 0 if the driver will not say.
///
/// `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE` on the context's device. cuda-oxide keeps
/// its own `device_attribute` helper private and exposes no L2 accessor, so this
/// goes through the raw bindings the same way [`device_memory`] does.
#[cfg(feature = "ref-loc-buckets")]
fn l2_cache_bytes() -> u64 {
    let mut dev: i32 = 0;
    // SAFETY: driver query against the current context; `dev` is a local.
    unsafe {
        if cuda_core::sys::cuCtxGetDevice(&mut dev) != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
            return 0;
        }
    }
    device_attribute(
        dev,
        cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
    )
    .map(|bytes| bytes.max(0) as u64)
    .unwrap_or(0)
}

/// Human-readable window size for a bucket shift. Shifts below 20 (a small
/// reference can legally floor as low as 16, i.e. 64 KiB) are sub-MiB, and
/// `bytes >> 20` there would be 0 — worse, computing `1u64 << (shift - 20)`
/// underflows the `u32` subtraction and wraps into a nonsense multi-exabyte
/// figure, which is the bug this exists to avoid.
#[cfg(feature = "ref-loc-buckets")]
fn window_desc(shift: u32) -> String {
    let bytes = 1u64 << shift;
    if shift >= 20 {
        format!("{} MiB", bytes >> 20)
    } else {
        format!("{} KiB", bytes >> 10)
    }
}

/// `HSPZ_BUCKET_ORDER_CHECK=k`: verify the permutation and the restored
/// survivor order on the first `k` chunks (`k` defaults to 3, `0` is off).
#[cfg(feature = "ref-loc-buckets")]
fn order_check_limit() -> u32 {
    match std::env::var("HSPZ_BUCKET_ORDER_CHECK") {
        Ok(v) => v.trim().parse::<u32>().unwrap_or(3),
        Err(_) => 0,
    }
}

/// Reference bytes the gate's own streams keep resident alongside the window:
/// one chunk of 32 anchors per resident warp (~3.7k warps x 256 B on an L4).
#[cfg(feature = "ref-loc-buckets")]
const BUCKET_STREAM_BYTES: u64 = 1 << 20;

/// Round 83 round 2: reference-address bucket geometry AND on/off policy for
/// `ref-loc-buckets`. Pure over its inputs (cycle 4) so the decision is unit
/// testable without a device context or env vars.
///
/// Picks the LARGEST window whose reference bytes, plus the sorted bitmask
/// (`max_hits/8`) and the gate's own resident streams, still fit three quarters
/// of `l2_bytes` — a bigger window means fewer buckets and a cheaper scatter,
/// and round 2's bitmask made the flag term 32x smaller, so there is room the
/// byte-flag version did not have. Two thirds was the round-83 setting; the L4
/// (48 MiB) then chose 16 MiB, and forced 32 MiB measured 2.8% less GPU busy
/// on W1 unit 0 in disjoint reversed pairs (Delta 53956614, 2026-09-09), so the
/// budget is 3/4: the L4 lands on 32 MiB, the 4090 (72 MiB) and T4 (4 MiB)
/// choices are unchanged. `B <= 32` is a hard kernel invariant of the ballot
/// loop, so the search never goes below the smallest legal shift.
///
/// If even that smallest legal window does not fit the budget (a small-L2
/// card), or `l2_bytes` is 0 (the L2 attribute is unavailable), a bucket
/// scatter cannot stay L2-resident and is a pure loss over the byte-flag
/// path: the third element is `false` and the returned shift/count are still
/// the smallest legal ones, so a caller that forces the pass on anyway has
/// somewhere sane to run.
///
/// `planes` (round 106): under the score-gate sieve the gate's random reads go
/// to the 2-bit planes, a quarter of a byte per base, and its keep bits are
/// written in order. A window of `2^s` bases then costs `2^s / 4` bytes and the
/// `max_hits/8` term drops out, so the window can be four times longer in bases:
/// fewer buckets and a cheaper pass. The budget is half of the L2 rather than
/// three quarters: on an A100 (40 MiB) 16 MiB of planes measured 3% less GPU
/// busy than 32 MiB. That A100's byte rule switched the pass off altogether at
/// production batch sizes; with this one it takes unit 0's sieve gate from
/// 16.0 to 6.8 s for a 3.6 s pass.
#[cfg(feature = "ref-loc-buckets")]
fn ref_bucket_policy(ref_len: u32, max_hits: u32, l2_bytes: u64, planes: bool) -> (u32, u32, bool) {
    let buckets = |s: u32| (ref_len >> s) + 1;
    // `ref_len` is a u32, so `ref_len >> 31 <= 1` and this always lands.
    let min_shift = (16..=31).find(|&s| buckets(s) <= 32).unwrap_or(31);
    let fits = |s: u32| {
        if planes {
            (1u64 << s) / 4 + BUCKET_STREAM_BYTES <= l2_bytes / 2
        } else {
            (1u64 << s) + u64::from(max_hits) / 8 + BUCKET_STREAM_BYTES <= (l2_bytes / 4) * 3
        }
    };
    match (min_shift..=31).rev().find(|&s| l2_bytes > 0 && fits(s)) {
        Some(shift) => (shift, buckets(shift), true),
        None => (min_shift, buckets(min_shift), false),
    }
}

/// `HSPZ_REF_BUCKET_SHIFT` pins the window (`16..=31`, i.e. 64 KiB..2 GiB) and
/// forces the pass on at that shift (so a small-L2 card's penalty can still be
/// measured); otherwise the geometry AND the on/off call both come from
/// [`ref_bucket_policy`].
#[cfg(feature = "ref-loc-buckets")]
fn ref_bucket_geometry(
    ref_len: u32,
    max_hits: u32,
    l2_bytes: u64,
    planes: bool,
) -> Result<(u32, u32, bool), String> {
    let buckets = |s: u32| (ref_len >> s) + 1;
    // `ref_len` is a u32, so `ref_len >> 31 <= 1` and this always lands.
    let min_shift = (16..=31).find(|&s| buckets(s) <= 32).unwrap_or(31);
    if let Ok(v) = std::env::var("HSPZ_REF_BUCKET_SHIFT") {
        let shift: u32 = v
            .trim()
            .parse()
            .map_err(|_| format!("HSPZ_REF_BUCKET_SHIFT={v:?} is not an integer"))?;
        if !(16..=31).contains(&shift) {
            return Err(format!(
                "HSPZ_REF_BUCKET_SHIFT={shift} outside the supported range 16..=31"
            ));
        }
        if buckets(shift) > 32 {
            return Err(format!(
                "HSPZ_REF_BUCKET_SHIFT={shift} needs {} buckets for a {ref_len} bp reference; \
the ballot loop allows at most 32 (use shift >= {min_shift})",
                buckets(shift),
            ));
        }
        return Ok((shift, buckets(shift), true));
    }
    let (shift, n_buckets, fits) = ref_bucket_policy(ref_len, max_hits, l2_bytes, planes);
    assert!(
        n_buckets <= 32,
        "ref bucket count {n_buckets} exceeds the 32-wide ballot loop"
    );
    Ok((shift, n_buckets, fits))
}

/// Copies the `[first, first+len)` range of a grown-once device buffer.
///
/// `DeviceBuffer::to_host_vec` copies its full high-water allocation, which is
/// both wasteful and makes audit transport accounting lie after truncation.
/// The range form keeps the over-cap chunk walk sparse: only the <=256-element
/// blocks holding a boundary cross the bus, on the caller's stream so each
/// copy is ordered after the queued `add_block_offsets`. The caller supplies
/// `host` so one small Engine-owned Vec is reused instead of allocating a
/// fresh pageable Vec per fetch.
fn copy_range_into<T: DeviceCopy>(
    buf: &DeviceBuffer<T>,
    stream: &CudaStream,
    first: usize,
    len: usize,
    host: &mut Vec<T>,
) -> Result<(), DriverError> {
    let end = first.checked_add(len).expect("device range end overflow");
    assert!(
        end <= buf.len(),
        "device range {first}..{end} exceeds allocation {}",
        buf.len()
    );
    host.clear();
    if len == 0 {
        return Ok(());
    }
    host.reserve(len);
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .expect("device range byte overflow");
    let byte_off = (first as u64)
        .checked_mul(std::mem::size_of::<T>() as u64)
        .expect("device range byte overflow") as cuda_core::sys::CUdeviceptr;
    // SAFETY: `host` owns writable capacity for `len` T values, `buf`
    // covers at least the `[first, first+len)` range, and the
    // synchronization completes the transfer before the vector is exposed
    // as initialized.
    // The copy into pageable memory waits for the stream itself, so it is timed
    // with the synchronization that follows it.
    host_trace::timed(&host_trace::BLOCKED_NS, || -> Result<(), DriverError> {
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                host.as_mut_ptr(),
                buf.cu_deviceptr() + byte_off,
                bytes,
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()
    })?;
    // SAFETY: the completed D2H copy initialized exactly `len` elements.
    unsafe { host.set_len(len) };
    Ok(())
}

/// Round 87 (blocks): runtime on/off self-calibration for `ref-loc-buckets`.
///
/// The L2-fit guard is not sufficient (a win on power-capped cards, a loss
/// on a 4090), so `HSPZ_REF_BUCKETS=auto` compares the two production paths
/// over settled clocks. The F1 win is clock-mediated — less DRAM traffic
/// lets the SM clock rise — but the power governor settles over seconds, so
/// a per-chunk pair sees only the pass cost, not the clock gain. Each path
/// therefore runs for a LONG block of chunks (>= T_BLOCK of gate span),
/// the first T_SETTLE of each block is ignored, and adjacent OFF/ON blocks
/// are compared on settled ns/hit. Every chunk in a block RUNS the block's
/// path in production: this is a timing policy, not an algorithm change,
/// and both paths produce identical bytes.
#[cfg(feature = "ref-loc-buckets")]
const BUCKET_BLOCKS: usize = 6;

/// Warm-up: the first 4 eligible chunks run OFF untimed (kernel loading,
/// first reservations) before block 0 starts.
#[cfg(feature = "ref-loc-buckets")]
const BUCKET_WARM_CHUNKS: u32 = 4;

/// Block length and settle window in ms of gate span. Overridable for tests
/// via `HSPZ_REF_BUCKETS_BLOCK_MS` / `HSPZ_REF_BUCKETS_SETTLE_MS`
/// (debug-only: a whole ZLUDA run is only ~100 chunks, so the defaults
/// would never close a block there).
#[cfg(feature = "ref-loc-buckets")]
fn bucket_block_ms() -> f64 {
    std::env::var("HSPZ_REF_BUCKETS_BLOCK_MS")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|&v| v > 0.0)
        .unwrap_or(3_000.0)
}

/// See [`bucket_block_ms`].
#[cfg(feature = "ref-loc-buckets")]
fn bucket_settle_ms() -> f64 {
    std::env::var("HSPZ_REF_BUCKETS_SETTLE_MS")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|&v| v >= 0.0)
        .unwrap_or(1_000.0)
}

/// One closed block: the settled ns/hit of a single path over its settled
/// chunks. `hits`/`span_ms` are the settled hits and the whole-block span;
/// the settled span is recoverable as `ns_per_hit * hits / 1e6`.
#[cfg(feature = "ref-loc-buckets")]
#[derive(Clone, Debug, PartialEq)]
struct BlockSample {
    on: bool,
    ns_per_hit: f64,
    chunks: u32,
    settled_chunks: u32,
    hits: u64,
    span_ms: f64,
}

/// Ledger fields for the autotune decision after the 6th block: the median
/// adjacent-pair ratio, the 3 ratios, and the per-path settled ns/hit.
#[cfg(feature = "ref-loc-buckets")]
#[derive(Debug, PartialEq)]
struct BlockDecision {
    use_on: bool,
    median: f64,
    ratios: Vec<f64>,
    off_ns: Vec<f64>,
    on_ns: Vec<f64>,
}

/// A block closed without a usable settled comparison: `span_ms >=
/// 2 * T_BLOCK` with fewer than 4 settled chunks (e.g. a tail of small
/// chunks). Exactness-neutral: the block ran its production path, only
/// the accounting is discarded.
#[cfg(feature = "ref-loc-buckets")]
#[derive(Clone, Debug, PartialEq)]
struct InconclusiveBlock {
    k: usize,
    on: bool,
    span_ms: f64,
    settled_chunks: u32,
}

/// `Auto` phase: untimed warm-up, then alternating OFF, ON, OFF, ON, OFF, ON
/// blocks (3 per path). `k` is the block index and always equals the number
/// of blocks closed so far.
#[cfg(feature = "ref-loc-buckets")]
#[derive(Debug, PartialEq)]
enum AutoPhase {
    Warm {
        n: u32,
    },
    Block {
        on: bool,
        k: usize,
        span_ms: f64,
        settled_span_ms: f64,
        settled_hits: u64,
        chunks: u32,
        settled_chunks: u32,
    },
}

/// Per-Engine bucket state: an explicit override (`1`/`0`, or geometry
/// auto-off forcing OFF before any block), the block-alternation phase, or
/// the committed decision.
#[cfg(feature = "ref-loc-buckets")]
#[derive(Debug, PartialEq)]
enum BucketMode {
    Forced(bool),
    Auto {
        phase: AutoPhase,
        blocks: Vec<BlockSample>,
    },
    Decided(bool),
}

/// `HSPZ_REF_BUCKETS=auto|1|0`; unset means `auto` (round 87): the first
/// chunks of every engine alternate OFF/ON in blocks of >= 3 s of gate span,
/// each block's first second is discarded (power-governor ramp), and the
/// engine commits to ON iff the median settled ON/OFF ns-per-hit ratio over
/// three adjacent pairs is below 0.97. Measured: L4 unit 0 0.941 (forced
/// arms 0.934) -> on; RTX 4090 unit 0 1.063-1.069 (forced ~1.05-1.08) -> off.
/// Output bytes are identical in every mode. Any other value is a clear
/// error. A pinned `HSPZ_REF_BUCKET_SHIFT` forces the pass on under `auto`
/// (only `HSPZ_REF_BUCKETS=0` opts out); cards whose L2 cannot hold the
/// smallest legal window (T4) force OFF before any trial.
#[cfg(feature = "ref-loc-buckets")]
enum BucketRequest {
    /// `HSPZ_REF_BUCKETS=1|0`: exactly that path, geometry notwithstanding.
    Forced(bool),
    /// `HSPZ_REF_BUCKETS=auto` or unset: alternate blocks and decide, if the
    /// window fits (the T4 guard forces OFF first).
    Auto,
}

#[cfg(feature = "ref-loc-buckets")]
fn parse_bucket_env() -> Result<BucketRequest, String> {
    match std::env::var("HSPZ_REF_BUCKETS").as_deref() {
        Err(_) => Ok(BucketRequest::Auto),
        Ok("auto") => Ok(BucketRequest::Auto),
        Ok("1") => Ok(BucketRequest::Forced(true)),
        Ok("0") => Ok(BucketRequest::Forced(false)),
        Ok(other) => Err(format!("HSPZ_REF_BUCKETS={other:?} is not one of auto|1|0")),
    }
}

/// Debug-only escape for verification: `HSPZ_REF_BUCKETS_TRIAL_ANYWAY=1`
/// makes `auto` run its blocks even when the geometry says the window
/// does not fit (e.g. ZLUDA's 1 MiB L2, which otherwise forces OFF before
/// any block). Never set in production; the decision it reaches on a
/// small-L2 card is meaningless, only the ledger shape and byte-identity
/// matter.
#[cfg(feature = "ref-loc-buckets")]
fn bucket_trial_anyway() -> bool {
    matches!(
        std::env::var("HSPZ_REF_BUCKETS_TRIAL_ANYWAY").as_deref(),
        Ok("1")
    )
}

/// A chunk counts as settled only if its hit count reaches a quarter of
/// the cap. Tails (and ineligible chunks generally) run in the current
/// block's path but never enter the settled totals.
#[cfg(feature = "ref-loc-buckets")]
fn bucket_trial_eligible(iter_hits: u32, max_hits: u32) -> bool {
    iter_hits >= max_hits / 4
}

/// Median of per-block on/off ratios (mean of the two middles when even).
#[cfg(feature = "ref-loc-buckets")]
fn median_ns_per_hit(v: &[f64]) -> f64 {
    debug_assert!(!v.is_empty());
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Ties and small margins go to the production default (OFF) path: ON wins
/// only with a >= 3% median margin on the per-block on/off ratios.
#[cfg(feature = "ref-loc-buckets")]
fn decide_bucket_autotune(ratios: &[f64]) -> bool {
    median_ns_per_hit(ratios) < 0.97
}

/// Adjacent-pair on/off ratios over closed blocks, which must read
/// OFF, ON, OFF, ON, OFF, ON: r_k = ns(ON block k) / ns(OFF block k).
#[cfg(feature = "ref-loc-buckets")]
fn block_pair_ratios(blocks: &[BlockSample]) -> Vec<f64> {
    debug_assert_eq!(blocks.len(), BUCKET_BLOCKS);
    let mut ratios = Vec::with_capacity(BUCKET_BLOCKS / 2);
    for pair in blocks.chunks_exact(2) {
        debug_assert!(!pair[0].on && pair[1].on);
        ratios.push(pair[1].ns_per_hit / pair[0].ns_per_hit);
    }
    ratios
}

#[cfg(feature = "ref-loc-buckets")]
impl BucketMode {
    /// Production path for a chunk: the pinned path for Forced/Decided, the
    /// warm-up OFF path or the current block's path for Auto. An engine that
    /// ends before 6 blocks simply stays in Auto on its current path.
    fn production_path(&self) -> bool {
        match self {
            BucketMode::Forced(on) | BucketMode::Decided(on) => *on,
            BucketMode::Auto { phase, .. } => match phase {
                AutoPhase::Warm { .. } => false,
                AutoPhase::Block { on, .. } => *on,
            },
        }
    }

    /// Advances the autotune state for this chunk and reports whether its
    /// gate+count span feeds the block accounting. The production path
    /// itself comes from [`production_path`](Self::production_path): warm
    /// chunks (the first 4 eligible) run OFF untimed, block chunks run the
    /// block's path timed, and Forced/Decided chunks never time.
    fn auto_chunk(&mut self, iter_hits: u32, max_hits: u32) -> bool {
        match self {
            BucketMode::Forced(_) | BucketMode::Decided(_) => false,
            BucketMode::Auto { phase, .. } => match phase {
                AutoPhase::Warm { n } => {
                    if bucket_trial_eligible(iter_hits, max_hits) {
                        *n += 1;
                        if *n >= BUCKET_WARM_CHUNKS {
                            *phase = AutoPhase::Block {
                                on: false,
                                k: 0,
                                span_ms: 0.0,
                                settled_span_ms: 0.0,
                                settled_hits: 0,
                                chunks: 0,
                                settled_chunks: 0,
                            };
                        }
                    }
                    false
                }
                AutoPhase::Block { .. } => true,
            },
        }
    }

    /// Commits one timed chunk span to the current block. Every chunk's span
    /// joins `span_ms`; once `span_ms` has passed `t_settle`, eligible
    /// chunks also join the settled totals. A block closes when `span_ms >=
    /// t_block` AND `settled_chunks >= 4`, recording
    /// `ns_per_hit = settled_span_ms * 1e6 / settled_hits`. Returns the
    /// closed block (index + sample) plus, after the 6th block, the decision
    /// (`Decided(true)` iff the median adjacent-pair ratio is < 0.97).
    ///
    /// Inconclusive close (exactness-neutral): a block that reaches
    /// `span_ms >= 2 * t_block` with fewer than 4 settled chunks (a tail of
    /// small chunks) can never decide. An inconclusive OFF block is dropped
    /// and a fresh OFF block starts at the same index; an inconclusive ON
    /// block ends the trial as `Decided(false)`. Reported as the third
    /// element; conclusive closes return it as `None`.
    fn commit_block_span(
        &mut self,
        span_ms: f64,
        iter_hits: u32,
        max_hits: u32,
        t_block: f64,
        t_settle: f64,
    ) -> (
        Option<(usize, BlockSample)>,
        Option<BlockDecision>,
        Option<InconclusiveBlock>,
    ) {
        let BucketMode::Auto { phase, blocks } = self else {
            return (None, None, None);
        };
        let AutoPhase::Block {
            on,
            k,
            span_ms: acc,
            settled_span_ms,
            settled_hits,
            chunks,
            settled_chunks,
        } = phase
        else {
            return (None, None, None);
        };
        debug_assert_eq!(*k, blocks.len());
        let settled_before = *acc >= t_settle;
        *acc += span_ms;
        *chunks += 1;
        if settled_before && bucket_trial_eligible(iter_hits, max_hits) {
            *settled_span_ms += span_ms;
            *settled_hits += u64::from(iter_hits);
            *settled_chunks += 1;
        }
        if *acc < t_block || *settled_chunks < 4 {
            if *acc >= 2.0 * t_block && *settled_chunks < 4 {
                let inc = InconclusiveBlock {
                    k: *k,
                    on: *on,
                    span_ms: *acc,
                    settled_chunks: *settled_chunks,
                };
                if *on {
                    *self = BucketMode::Decided(false);
                } else {
                    *phase = AutoPhase::Block {
                        on: false,
                        k: *k,
                        span_ms: 0.0,
                        settled_span_ms: 0.0,
                        settled_hits: 0,
                        chunks: 0,
                        settled_chunks: 0,
                    };
                }
                return (None, None, Some(inc));
            }
            return (None, None, None);
        }
        debug_assert!(*settled_hits > 0);
        let closed_k = *k;
        let closed = BlockSample {
            on: *on,
            ns_per_hit: *settled_span_ms * 1e6 / (*settled_hits as f64),
            chunks: *chunks,
            settled_chunks: *settled_chunks,
            hits: *settled_hits,
            span_ms: *acc,
        };
        blocks.push(closed.clone());
        if blocks.len() < BUCKET_BLOCKS {
            *phase = AutoPhase::Block {
                on: !closed.on,
                k: closed_k + 1,
                span_ms: 0.0,
                settled_span_ms: 0.0,
                settled_hits: 0,
                chunks: 0,
                settled_chunks: 0,
            };
            return (Some((closed_k, closed)), None, None);
        }
        let ratios = block_pair_ratios(blocks);
        let use_on = decide_bucket_autotune(&ratios);
        let decision = BlockDecision {
            use_on,
            median: median_ns_per_hit(&ratios),
            ratios,
            off_ns: blocks.iter().step_by(2).map(|b| b.ns_per_hit).collect(),
            on_ns: blocks
                .iter()
                .skip(1)
                .step_by(2)
                .map(|b| b.ns_per_hit)
                .collect(),
        };
        *self = BucketMode::Decided(use_on);
        (Some((closed_k, closed)), Some(decision), None)
    }

    /// Closed blocks so far, if the engine is still in `Auto` (`None` once
    /// forced or decided). The engine-end ledger uses this to say why there
    /// is no decision.
    fn undecided_blocks(&self) -> Option<usize> {
        match self {
            BucketMode::Auto { blocks, .. } => Some(blocks.len()),
            BucketMode::Forced(_) | BucketMode::Decided(_) => None,
        }
    }
}

/// Copies only the initialized prefix of a grown-once device buffer.
/// `DeviceBuffer::to_host_vec` copies its full high-water allocation, which is
/// both wasteful and makes audit transport accounting lie after truncation.
fn copy_prefix<T: DeviceCopy>(
    buf: &DeviceBuffer<T>,
    stream: &CudaStream,
    len: usize,
) -> Result<Vec<T>, DriverError> {
    let mut host = Vec::with_capacity(len);
    copy_range_into(buf, stream, 0, len, &mut host)?;
    Ok(host)
}

/// An event used only for ordering, never for timing — cheaper to record, and
/// `elapsed_ms` on one would be a driver error, which is the point: these carry
/// dependencies, not measurements.
fn untimed_event(ctx: &Arc<CudaContext>) -> Result<CudaEvent, DriverError> {
    ctx.new_event(Some(
        cuda_core::sys::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING,
    ))
}

/// An event that can also be read with `elapsed_ms`.
fn timed_event(ctx: &Arc<CudaContext>) -> Result<CudaEvent, DriverError> {
    ctx.new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
}

/// Config mirroring the arguments of `InitializeProcessor`.
pub struct EngineConfig<'a> {
    pub index_table: &'a [u32],
    pub pos_table: &'a [u32],
    /// Reference block, already in the device alphabet.
    pub ref_seq: &'a [u8],
    pub sub_mat: &'a [i32],
    pub seed_size: u32,
    pub xdrop: i32,
    pub hspthresh: i32,
    pub noentropy: bool,
    /// Resolved hit cap. `0` still derives from *this* device for tests;
    /// production always passes a value resolved once from device 0.
    pub max_hits: u32,
    /// Physical hit allowance (`C >= H`). Semantic chunking still targets
    /// `max_hits`; capacity only decides success/failure of an over-target
    /// chunk walk. `0` defaults to `max_hits`.
    pub hit_capacity: u32,
    pub timing: bool,
    /// N8: `find_hsps` grid. `0` uses [`HSP_BLOCKS`], the ZLUDA
    /// optimum. Runtime rather than `const` so the L4 sweep needs no rebuild.
    pub hsp_blocks: u32,
    /// Round 122: `Some` asks for the transition fold ([`FoldRequest`]). The
    /// engine decides ([`Engine::folded`]).
    #[cfg_attr(
        not(all(
            feature = "find-hits-groups",
            feature = "ref-loc-buckets",
            feature = "gate-sieve"
        )),
        allow(dead_code)
    )]
    pub fold: Option<FoldRequest>,
}

/// Round 122, the transition fold. A query position's seeds are its k-mer and
/// that k-mer with one care position changed by a transition; their position
/// lists are disjoint, so one table in which every k-mer also holds its
/// variants' positions gives a position's hits from one seed. An engine that
/// has it (`Engine::folded`) is seeded without transitions and finds the same
/// hits: a thirteenth of the seeds, slices thirteen times as long.
///
/// A batch is the same set of hits either way, but a batch of `max_hits` hits
/// or more is cut into chunks at seeds of the plain order, and the output
/// depends on the cut (the dedup runs per chunk). Such a batch comes back
/// `oversized` and the caller sends its plain seeds
/// ([`Engine::plain_next`]); the engine keeps the plain tables for it.
// The fold needs the three features of the group line; without them a request is carried and not read.
#[cfg_attr(
    not(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    )),
    allow(dead_code)
)]
#[derive(Clone, Copy, Debug)]
pub struct FoldRequest {
    /// Bit `t` set for every care position that admits a transition.
    pub mask: u32,
    /// Device bytes that one hit of capacity costs
    /// (`plan::device_bytes_per_hit`). The folded tables are paid for with
    /// capacity above `max_hits`, which only a batch cut into chunks can need:
    /// the fold is built when `max_hits` still fits beside it, and dropped for
    /// the rest of the reference the first time a chunk needs more than what
    /// is left, or when more than a quarter of the batches come back
    /// ([`Engine::folded`] then turns false).
    pub bytes_per_hit: u64,
    /// `HSPZ_FOLD=check`: the device's tables against the host's
    /// (`seed::fold_tables`), every k-mer.
    pub check: bool,
}

/// Round 122: from this many batches sent again, a folded engine whose
/// batches come back more than one time in four drops its fold.
const FOLD_AGAIN_MIN: u32 = 16;

/// Round 122: what the fold did over the run.
pub mod fold_totals {
    use std::sync::atomic::AtomicU64;
    #[cfg_attr(
        not(all(
            feature = "find-hits-groups",
            feature = "ref-loc-buckets",
            feature = "gate-sieve"
        )),
        allow(dead_code)
    )]
    pub static BINS: AtomicU64 = AtomicU64::new(0);
    pub static BATCHES: AtomicU64 = AtomicU64::new(0);
    pub static REISSUED: AtomicU64 = AtomicU64::new(0);
    pub static RELEASED: AtomicU64 = AtomicU64::new(0);
    /// A reference of this run dropped its fold because its batches are cut at
    /// `--max-hits`: no later reference builds one.
    pub static CUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
}

impl Engine {
    pub fn new(
        ctx: &Arc<CudaContext>,
        cfg: EngineConfig<'_>,
        phases: &mut Phases,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = ctx.default_stream();

        let t = Instant::now();
        let module = kernels::device::load(ctx)?;
        phases.add("module load / JIT", t.elapsed());

        let max_hits = if cfg.max_hits > 0 {
            cfg.max_hits
        } else {
            default_max_hits(ctx)
        };
        let hsp_blocks = if cfg.hsp_blocks > 0 {
            cfg.hsp_blocks
        } else {
            HSP_BLOCKS
        };
        let hit_capacity = if cfg.hit_capacity > 0 {
            cfg.hit_capacity
        } else {
            max_hits
        };
        validate_hit_config(max_hits, hit_capacity, hsp_blocks)?;
        // Cycle 4 (round 87): `HSPZ_REF_BUCKETS=0`/`=1` are explicit overrides
        // and win over the policy either way; `=auto` runs alternating OFF/ON
        // blocks of the production path and commits to the faster settled
        // ns/hit, unless the geometry auto-offs first (a pinned
        // `HSPZ_REF_BUCKET_SHIFT` always reports "fits", which is how it
        // forces the pass on). `HSPZ_REF_BUCKETS_TRIAL_ANYWAY=1` (debug-only)
        // lifts the geometry auto-off under `auto` so the blocks can be
        // exercised where the window does not fit.
        // Under the sieve the bucket window is sized for the packed planes.
        // Whether the sieve runs is settled per unit (IUPAC codes in a query
        // switch it off); what can be known here is whether the scoring allows it.
        #[cfg(feature = "gate-sieve")]
        let bucket_planes = sieve_request().0
            && sieve::classes(cfg.sub_mat, cfg.xdrop, false, false, sieve_masked()).is_ok();
        #[cfg(all(feature = "ref-loc-buckets", not(feature = "gate-sieve")))]
        let bucket_planes = false;
        #[cfg(feature = "ref-loc-buckets")]
        let (bucket_shift, n_buckets, bucket_mode) = {
            let l2 = l2_cache_bytes();
            let (shift, n, fits) =
                ref_bucket_geometry(cfg.ref_seq.len() as u32, max_hits, l2, bucket_planes)?;
            let req = parse_bucket_env().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
            // A pinned shift is an explicit request: force ON under `auto`.
            // Geometry auto-off still wins over everything (a pinned shift
            // reports fits, so this arm only fires for the policy path),
            // unless the debug-only TRIAL_ANYWAY escape is set.
            let pinned = std::env::var("HSPZ_REF_BUCKET_SHIFT").is_ok();
            let anyway = bucket_trial_anyway();
            let auto = match req {
                BucketRequest::Forced(on) => BucketMode::Forced(on),
                // The geometry guard is unconditional for the default path; the
                // debug escape lifts it only for `auto` trials.
                BucketRequest::Auto if !fits && !anyway => BucketMode::Forced(false),
                BucketRequest::Auto if pinned && !anyway => BucketMode::Forced(true),
                // Under the sieve a plane-sized window that fits is simply on,
                // unless `auto` was asked for by name. The trial needs 18 s of
                // gate per engine, a quarter of a reference bin's gate at sieve
                // speed, and only counts chunks of at least a quarter of
                // `max_hits`, which an 80 GB card's production batches never
                // reach: there it would stay on the plain path for ever.
                BucketRequest::Auto
                    if bucket_planes && !anyway && std::env::var("HSPZ_REF_BUCKETS").is_err() =>
                {
                    BucketMode::Forced(true)
                }
                BucketRequest::Auto => BucketMode::Auto {
                    phase: AutoPhase::Warm { n: 0 },
                    blocks: Vec::new(),
                },
            };
            let on = match auto {
                BucketMode::Forced(on) => on,
                BucketMode::Auto { .. } => fits,
                BucketMode::Decided(_) => unreachable!("fresh mode is never decided"),
            };
            if cfg.timing {
                if on {
                    eprintln!(
                        "  ref buckets: on — shift {} ({} window{}), {} buckets, L2 {} MiB ({})",
                        shift,
                        window_desc(shift),
                        if bucket_planes {
                            format!(", {} of packed planes", window_desc(shift - 2))
                        } else {
                            String::new()
                        },
                        n,
                        l2 >> 20,
                        if l2 > 0 { "device attr" } else { "unavailable" },
                    );
                } else {
                    // A forced OFF says so; the geometry text is only the reason when
                    // the policy itself turned the pass off (it used to be printed
                    // under HSPZ_REF_BUCKETS=0 even when the window fit).
                    let reason = if std::env::var("HSPZ_REF_BUCKETS").as_deref() == Ok("0") {
                        "forced by HSPZ_REF_BUCKETS=0".to_string()
                    } else if l2 == 0 {
                        "L2 attribute unavailable".to_string()
                    } else {
                        format!(
                            "{} smallest legal window exceeds the {} budget (L2 {} MiB)",
                            window_desc(shift),
                            if bucket_planes {
                                "half-L2 packed-plane"
                            } else {
                                "3/4-L2"
                            },
                            l2 >> 20,
                        )
                    };
                    eprintln!("  ref buckets: off — {reason}");
                }
            }
            (shift, n, auto)
        };

        let t = Instant::now();
        let engine = Engine {
            index_table: DeviceBuffer::from_host(&stream, cfg.index_table)?,
            pos_table: DeviceBuffer::from_host(&stream, cfg.pos_table)?,
            ref_seq: DeviceBuffer::from_host(&stream, cfg.ref_seq)?,
            // Reference-only. Every query bin — including the first —
            // enters through `swap_query`, so `query_swaps == work_units` holds
            // and no query is ever uploaded twice.
            query_seq: DeviceBuffer::from_host(&stream, &[] as &[u8])?,
            query_rc_seq: DeviceBuffer::from_host(&stream, &[] as &[u8])?,
            sub_mat: DeviceBuffer::from_host(&stream, cfg.sub_mat)?,
            ref_len: cfg.ref_seq.len() as u32,
            query_len: 0,
            seed_size: cfg.seed_size,
            xdrop: cfg.xdrop,
            hspthresh: cfg.hspthresh,
            gate_thresh: if cfg!(feature = "gate-int-keep") {
                keep_threshold(cfg.hspthresh)
            } else {
                cfg.hspthresh
            },
            #[cfg(feature = "gate-sieve")]
            sieve: GateSieve::new(&stream, cfg.sub_mat)?,
            noentropy: cfg.noentropy as u32,
            max_hits,
            hit_capacity,
            chunk_walk_full: matches!(std::env::var("HSPZ_CHUNK_WALK").as_deref(), Ok("full")),
            chunk_block_scratch: Vec::with_capacity(SCAN_BLOCK as usize),
            timing: cfg.timing,
            hsp_blocks,
            dump_raw: false,
            paired: None,
            fold_alt: None,
            fold_active: false,
            plain_once: false,
            fold_capacity: 0,
            fold_seen: 0,
            fold_again: 0,
            paired_multi: false,
            buf_seed_lo: unsafe { uninitialized(&stream, 0)? },
            collect_hit_stats: false,
            census: crate::census::SurvivorAudit::enabled()
                .then(crate::census::SurvivorAudit::default),
            phases: Phases::new(),
            hit_stats: HitStats::default(),
            #[cfg(feature = "counters")]
            hsp_stats: HspStats::default(),
            #[cfg(feature = "counters")]
            groups: crate::hsp::Groups::default(),
            launches: 0,
            peak_used: 0,
            // Grown to the first batch's size on use.
            buf_hsp: unsafe { uninitialized(&stream, 0)? },
            buf_done: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "dense-anchors")]
            buf_anchor: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "dense-anchors")]
            buf_flags: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "dense-anchors")]
            buf_survivors: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "ref-loc-buckets")]
            buf_sorted_anchor: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "ref-loc-buckets")]
            buf_sorted_idx: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "ref-loc-buckets")]
            buf_bucket_counts: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "ref-loc-buckets")]
            buf_flags_bits: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "ref-loc-buckets")]
            bucket_planes,
            #[cfg(all(
                feature = "find-hits-groups",
                feature = "ref-loc-buckets",
                feature = "gate-sieve"
            ))]
            hits_bucketed: hits_bucketed_request(),
            #[cfg(all(
                feature = "find-hits-groups",
                feature = "ref-loc-buckets",
                feature = "gate-sieve"
            ))]
            buf_bucket_bounds: unsafe { uninitialized(&stream, 0)? },
            #[cfg(all(
                feature = "find-hits-groups",
                feature = "ref-loc-buckets",
                feature = "gate-sieve"
            ))]
            buf_seed_counts: unsafe { uninitialized(&stream, 0)? },
            #[cfg(all(
                feature = "find-hits-groups",
                feature = "ref-loc-buckets",
                feature = "gate-sieve"
            ))]
            buf_bucket_nibbles: unsafe { uninitialized(&stream, 0)? },
            #[cfg(all(
                feature = "find-hits-groups",
                feature = "ref-loc-buckets",
                feature = "gate-sieve"
            ))]
            buf_seed_nib: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "ref-loc-buckets")]
            bucket_match: bucket_match_request(),
            #[cfg(all(feature = "ref-loc-buckets", feature = "gate-sieve"))]
            survivor_slots: survivor_slots_request(),
            #[cfg(feature = "find-hits-groups")]
            find_hits_groups: find_hits_groups_request(),
            #[cfg(feature = "ref-loc-buckets")]
            bucket_shift,
            #[cfg(feature = "ref-loc-buckets")]
            n_buckets,
            #[cfg(feature = "ref-loc-buckets")]
            bucket_mode,
            #[cfg(feature = "ref-loc-buckets")]
            block_span_start: None,
            #[cfg(feature = "ref-loc-buckets")]
            block_span_end: None,
            #[cfg(feature = "ref-loc-buckets")]
            bucket_eng_id: {
                static ENG_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                ENG_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            },
            persistent_seed_buffers: false,
            seed_slots: [unsafe { uninitialized(&stream, 0)? }, unsafe {
                uninitialized(&stream, 0)?
            }],
            seed_len: [0, 0],
            buf_hit_num: unsafe { uninitialized(&stream, 0)? },
            seed_shape: unsafe { uninitialized(&stream, 0)? },
            buf_seed_kmer: unsafe { uninitialized(&stream, 0)? },
            buf_seed_offsets: unsafe { uninitialized(&stream, 0)? },
            scr_hit_sums: unsafe { uninitialized(&stream, 0)? },
            scr_hit_offsets: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "ref-loc-buckets")]
            scr_run_sums: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "dense-anchors")]
            scr_surv_sums: unsafe { uninitialized(&stream, 0)? },
            #[cfg(feature = "dense-anchors")]
            scr_surv_offsets: unsafe { uninitialized(&stream, 0)? },
            scr_done_sums: unsafe { uninitialized(&stream, 0)? },
            scr_done_offsets: unsafe { uninitialized(&stream, 0)? },
            #[cfg(not(feature = "counters"))]
            scr_stats: DeviceBuffer::<u64>::zeroed(&stream, 1)?,
            buf_reduced: unsafe { uninitialized(&stream, 0)? },
            mem_epoch_seen: u64::MAX,
            pinned_slots: Vec::new(),
            pinned_ok: false,
            last_end: None,
            last_name: "",
            gap_ms: 0.0,
            gap_n: 0,
            gap_max: 0.0,
            gap_max_pair: ("", ""),
            gap_pairs: Vec::new(),
            gap_long_n: 0,
            gap_long_ms: 0.0,
            device_seeds: false,
            reference_uploads: 1,
            query_swaps: 0,
            pending: Vec::new(),
            stage_syncs: 0,
            pipeline_syncs: 0,
            async_stages: false,
            copy_stream: ctx.new_stream()?,
            // `copy_ready` doubles as the end event for the overlapped-copy timing
            // row, and `elapsed_ms` refuses a DISABLE_TIMING handle, so this pair is
            // timed. `compute_done` is never measured.
            copy_ready: [timed_event(ctx)?, timed_event(ctx)?],
            compute_done: [untimed_event(ctx)?, untimed_event(ctx)?],
            copy_start: [None, None],
            async_seed_copy: false,
            seed_uploads: 0,
            seed_copy_stalls: 0,
            module,
            stream,
        };
        for s in 0..2 {
            engine.copy_ready[s].record(&engine.copy_stream)?;
            engine.compute_done[s].record(&engine.stream)?;
        }
        #[cfg(feature = "gate-sieve")]
        let mut engine = engine;
        #[cfg(feature = "gate-sieve")]
        engine.sieve_pack_reference(cfg.ref_seq)?;
        #[cfg(all(
            feature = "find-hits-groups",
            feature = "ref-loc-buckets",
            feature = "gate-sieve"
        ))]
        engine.build_bucket_tables()?;
        #[cfg(all(
            feature = "find-hits-groups",
            feature = "ref-loc-buckets",
            feature = "gate-sieve"
        ))]
        if let Some(req) = cfg.fold {
            engine.fold_tables(req, cfg.index_table, cfg.pos_table)?;
        }
        engine.stream.synchronize()?;
        engine.copy_stream.synchronize()?;
        phases.add("upload ref/query/tables", t.elapsed());
        Ok(engine)
    }

    /// `--paired`: uploads each pair's packed bounds and switches this engine to
    /// the pair-restricted hit kernels. `q_start` must be ascending; pair `p`
    /// owns query positions from `q_start[p]` and reference positions
    /// `[t_lo[p], t_hi[p])`.
    pub fn set_pairs(
        &mut self,
        q_start: &[u32],
        t_lo: &[u32],
        t_hi: &[u32],
    ) -> Result<(), DriverError> {
        self.paired = Some(PairedBounds {
            q_start: DeviceBuffer::from_host(&self.stream, q_start)?,
            t_lo: DeviceBuffer::from_host(&self.stream, t_lo)?,
            t_hi: DeviceBuffer::from_host(&self.stream, t_hi)?,
        });
        // The engine was built before its pairs were known, with the tables of
        // the group line (round 119). They count a k-mer's hits over the whole
        // reference; a pair takes a slice of them, so they must not be used.
        #[cfg(all(
            feature = "find-hits-groups",
            feature = "ref-loc-buckets",
            feature = "gate-sieve"
        ))]
        // SAFETY: empty buffers, never read.
        unsafe {
            self.buf_bucket_nibbles = uninitialized(&self.stream, 0)?;
            self.buf_bucket_bounds = uninitialized(&self.stream, 0)?;
        }
        Ok(())
    }

    /// Replaces only the query-side device state, keeping this reference bin's
    /// `index_table`, `pos_table`, `ref_seq` and `sub_mat` resident.
    ///
    /// This is what makes reference-bin reuse real. Constructing a fresh
    /// `Engine` per work unit would re-upload `index_table` (~67 MB) and
    /// `pos_table` (~1 GB for a chr1-sized bin at `--step 1`) for every query
    /// bin: a 12x12 plan would move ~150 GB extra over the bus, an order of
    /// magnitude more than the SeedTable stage this project just optimised from
    /// 11.5 s to 3.0 s.
    ///
    /// `reference_uploads` deliberately does not move here — see
    /// [`reference_uploads`](Self::reference_uploads).
    pub fn swap_query(&mut self, query_seq: &[u8], query_rc_seq: &[u8]) -> Result<(), DriverError> {
        // The old planes go before the new query arrives, so the swap never
        // holds more than the three query buffers the plan budgets for.
        #[cfg(feature = "gate-sieve")]
        self.sieve.release_query(&self.stream)?;
        self.query_seq = DeviceBuffer::from_host(&self.stream, query_seq)?;
        self.query_rc_seq = DeviceBuffer::from_host(&self.stream, query_rc_seq)?;
        self.query_len = query_seq.len() as u32;
        self.query_swaps += 1;
        #[cfg(feature = "gate-sieve")]
        self.sieve_pack_query(query_seq, query_rc_seq)?;
        Ok(())
    }

    /// How many times this engine uploaded a reference index.
    ///
    /// The executor asserts this equals the *visit* count, not the
    /// work-unit count. Counting host `SeedTable::build` calls alone would miss a
    /// refactor that rebuilt the `Engine` per pair, which is the regression the
    /// amendment is guarding against — so this counts device uploads.
    pub fn reference_uploads(&self) -> u32 {
        self.reference_uploads
    }

    /// Query swaps performed, i.e. work units run against the resident
    /// reference bin.
    pub fn query_swaps(&self) -> u32 {
        self.query_swaps
    }

    /// Hands out a pinned host staging buffer of at least `cap` elements,
    /// reusing one from the pool when possible.
    ///
    /// Pinned pages let the driver DMA straight out of host memory instead of
    /// staging a pageable copy, and they are the prerequisite for a genuinely
    /// async H->D. Allocation is the catch: page-locking is slow enough that
    /// doing it per pass costs more than the transfer it saves, so callers
    /// return buffers with [`give_pinned`](Self::give_pinned) and the engine
    /// keeps them alive.
    pub fn take_pinned(
        &mut self,
        cap: usize,
    ) -> Result<cuda_core::PinnedHostBuffer<u64>, DriverError> {
        if let Some(i) = self.pinned_slots.iter().position(|b| b.len() >= cap) {
            return Ok(self.pinned_slots.swap_remove(i));
        }
        cuda_core::PinnedHostBuffer::<u64>::zeroed(self.stream.context(), cap)
    }

    /// Returns a buffer from [`take_pinned`](Self::take_pinned) to the pool.
    pub fn give_pinned(&mut self, buf: cuda_core::PinnedHostBuffer<u64>) {
        self.pinned_slots.push(buf);
        self.pinned_ok = true;
    }

    /// Whether pinned staging actually engaged. False when the
    /// driver refused `cuMemHostAlloc` and the run fell back to pageable pages,
    /// which must not be reported as a pinned result.
    pub fn pinned_seeds_active(&self) -> bool {
        self.pinned_ok
    }

    /// The `max_hits` actually in force — either `--max-hits` or the value
    /// derived from device memory. Recorded in the JSON because it is
    /// load-bearing for final-HSP parity across devices with different VRAM.
    pub fn max_hits(&self) -> u32 {
        self.max_hits
    }

    /// Physical hit allowance in force (`C >= H`). Only decides
    /// success/failure of an over-target chunk walk, never output bytes.
    pub fn hit_capacity(&self) -> u32 {
        self.hit_capacity
    }

    /// The `find_hsps` grid in force (N8).
    pub fn hsp_blocks(&self) -> u32 {
        self.hsp_blocks
    }

    /// Mean wall-clock cost of one kernel launch, measured with an empty
    /// kernel. Under ZLUDA every launch pays PTX-to-HIP dispatch, so this is
    /// what tells kernel time apart from launch time.
    pub fn launch_overhead_ms(&self, iters: u32) -> Result<f64, DriverError> {
        let sink = DeviceBuffer::<u32>::zeroed(&self.stream, 1)?;
        let mut out = DeviceBuffer::<u32>::zeroed(&self.stream, 1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        // Warm the dispatch path before measuring it.
        for _ in 0..16 {
            // SAFETY: the kernel touches nothing; `sink` is present only so the
            // launch marshals an argument like a real one.
            unsafe { self.module.noop(&self.stream, cfg, &sink, &mut out)? };
        }
        self.stream.synchronize()?;

        let t = Instant::now();
        for _ in 0..iters {
            // SAFETY: as above.
            unsafe { self.module.noop(&self.stream, cfg, &sink, &mut out)? };
        }
        self.stream.synchronize()?;
        Ok(t.elapsed().as_secs_f64() * 1000.0 / iters as f64)
    }

    /// Stages one batch's seeds into device slot `slot`.
    ///
    /// Default: a blocking copy on the compute stream, exactly as before.
    /// `--async-seed-copy` (Phase 3): the copy goes on a second stream, ordered
    /// after `compute_done[slot]` so it cannot overwrite a buffer a queued kernel
    /// is still reading, and the compute stream later waits on `copy_ready[slot]`.
    /// The caller issues it one batch ahead, which is what gives the DMA something
    /// to hide behind.
    pub fn upload_seeds(&mut self, slot: usize, seeds: &[u64]) -> Result<(), DriverError> {
        let t = Instant::now();
        // SAFETY: fully overwritten by the copy below, and every kernel and copy
        // is bounded by `seed_len[slot]`, so spare capacity is never read.
        unsafe {
            reserve(
                &mut self.seed_slots[slot],
                &self.stream,
                seeds.len(),
                &mut self.pipeline_syncs,
            )?;
        }
        self.seed_len[slot] = seeds.len() as u32;
        if seeds.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::size_of_val(seeds);
        if self.async_seed_copy {
            self.copy_stream.wait(&self.compute_done[slot])?;
            if self.timing {
                let ev = match self.copy_start[slot].take() {
                    Some(ev) => ev,
                    None => self
                        .stream
                        .context()
                        .new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?,
                };
                ev.record(&self.copy_stream)?;
                self.copy_start[slot] = Some(ev);
            }
            // SAFETY: `bytes <= seed_slots[slot].len() * 8` by the reserve above.
            // The host slice must outlive the copy: the caller keeps one host slot
            // per in-flight batch and only reuses a slot after the batch that owned
            // it has finished computing, which required this copy to complete.
            unsafe {
                cuda_core::memory::memcpy_htod_async(
                    self.seed_slots[slot].cu_deviceptr(),
                    seeds.as_ptr(),
                    bytes,
                    self.copy_stream.cu_stream(),
                )?;
            }
            self.copy_ready[slot].record(&self.copy_stream)?;
            self.seed_uploads += 1;
        } else {
            // A persistent buffer is sized to the high-water batch, so the safe
            // `copy_from_host` (which requires exactly equal lengths) cannot be
            // used — copy the prefix explicitly instead. Same bytes, same ordering.
            //
            // SAFETY: as above; a synchronous copy cannot outlive the borrowed
            // host slice.
            unsafe {
                cuda_core::memory::memcpy_htod_sync(
                    self.seed_slots[slot].cu_deviceptr(),
                    seeds.as_ptr(),
                    bytes,
                )?;
            }
            self.absorb_sync()?;
            self.seed_uploads += 1;
        }
        self.phases.add("H->D seeds", t.elapsed());
        Ok(())
    }

    /// Round 122: waits for every seed upload that has been issued. An upload
    /// runs on its own stream and is ordered against the kernels that read its
    /// slot, not against a kernel that writes the slot: the device seeder,
    /// when a batch whose folded seeds were uploaded is seeded again.
    pub fn settle_uploads(&mut self) -> Result<(), DriverError> {
        host_trace::timed(&host_trace::BLOCKED_NS, || self.copy_stream.synchronize())
    }

    /// Generate one batch's exact compact seed stream on the device.
    pub fn generate_seeds(
        &mut self,
        slot: usize,
        rev: bool,
        range: (u32, u32),
        shape: &Shape,
        transitions: bool,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        if shape.size as u32 != self.seed_size {
            return Err("device seed shape does not match Engine seed size".into());
        }
        let span = range.1.saturating_sub(range.0);
        if span == 0 {
            self.seed_len[slot] = 0;
            return Ok(0);
        }
        if range.1.saturating_add(self.seed_size - 1) > self.query_len {
            return Err(format!(
                "device seed range {:?} exceeds query length {}",
                range, self.query_len
            )
            .into());
        }
        if self.seed_shape.is_empty() {
            let pos: Vec<u32> = shape.pos.iter().map(|&p| p as u32).collect();
            self.seed_shape = DeviceBuffer::from_host(&self.stream, &pos)?;
        }
        if self.seed_shape.len() != shape.kmer_size {
            return Err("device seed shape changed within one Engine".into());
        }

        let t = Instant::now();
        unsafe {
            reserve(
                &mut self.buf_seed_kmer,
                &self.stream,
                span as usize,
                &mut self.pipeline_syncs,
            )?;
            reserve(
                &mut self.buf_hit_num,
                &self.stream,
                span as usize,
                &mut self.pipeline_syncs,
            )?;
        }
        self.phases.add("alloc seed generation", t.elapsed());

        let per_pos = if transitions {
            1 + shape.kmer_size as u32
        } else {
            1
        };
        let query = if rev {
            &self.query_rc_seq
        } else {
            &self.query_seq
        };
        let stage = self.stage();
        unsafe {
            self.module.seed_kmers(
                &self.stream,
                elementwise(),
                query,
                &self.seed_shape,
                self.seed_size,
                range.0,
                span,
                per_pos,
                &mut self.buf_seed_kmer,
                &mut self.buf_hit_num,
            )?;
        }
        self.end_stage(stage, "seed k-mers")?;

        let blocks = span.div_ceil(SCAN_BLOCK);
        // SAFETY: `scan_blocks` writes one sum for every launched block.
        unsafe {
            reserve(
                &mut self.scr_hit_sums,
                &self.stream,
                blocks as usize,
                &mut self.pipeline_syncs,
            )?;
        }
        let stage = self.stage();
        unsafe {
            self.module.scan_blocks(
                &self.stream,
                LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (SCAN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut self.buf_hit_num,
                &mut self.scr_hit_sums,
                span,
            )?;
        }
        self.end_stage(stage, "seed scan blocks")?;

        let t = Instant::now();
        let mut sums = copy_prefix(&self.scr_hit_sums, &self.stream, blocks as usize)?;
        self.absorb_sync()?;
        let mut num_seeds = 0u32;
        for sum in &mut sums {
            let total = *sum;
            *sum = num_seeds;
            num_seeds += total;
        }
        self.seed_len[slot] = num_seeds;
        if num_seeds == 0 {
            self.phases.add("seed count round trip", t.elapsed());
            return Ok(0);
        }
        // SAFETY: the copy is queued before the kernel that reads it, and the
        // stream is drained again (the hit-count scan below) before `sums` goes.
        unsafe {
            upload(
                &mut self.buf_seed_offsets,
                &self.stream,
                &sums,
                &mut self.pipeline_syncs,
            )?;
        }
        self.phases.add("seed count round trip", t.elapsed());

        let stage = self.stage();
        unsafe {
            self.module.add_block_offsets(
                &self.stream,
                LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (SCAN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut self.buf_hit_num,
                &self.buf_seed_offsets,
                span,
            )?;
        }
        self.end_stage(stage, "seed add offsets")?;

        unsafe {
            reserve(
                &mut self.seed_slots[slot],
                &self.stream,
                num_seeds as usize,
                &mut self.pipeline_syncs,
            )?;
        }
        let stage = self.stage();
        unsafe {
            self.module.scatter_seeds(
                &self.stream,
                elementwise(),
                &self.buf_seed_kmer,
                &self.buf_hit_num,
                range.0,
                span,
                shape.kmer_size as u32,
                transitions as u32,
                &mut self.seed_slots[slot],
            )?;
        }
        self.end_stage(stage, "seed scatter")?;
        Ok(num_seeds)
    }

    /// Byte-for-byte seed oracle used only by the correctness build of the device seeder.
    #[cfg(feature = "device-seeds-check")]
    pub fn check_seed_bytes(
        &mut self,
        slot: usize,
        expected: &[u64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.seed_len[slot] as usize != expected.len() {
            return Err(format!(
                "device seed count differs: {} != {}",
                self.seed_len[slot],
                expected.len()
            )
            .into());
        }
        let got = self.seed_slots[slot].to_host_vec(&self.stream)?;
        self.absorb_sync()?;
        if let Some(i) = got[..expected.len()]
            .iter()
            .zip(expected)
            .position(|(a, b)| a != b)
        {
            return Err(format!(
                "device seed differs at {i}: {:016x} != {:016x}",
                got[i], expected[i]
            )
            .into());
        }
        Ok(())
    }

    /// One `SeedAndFilter(seed_offset_vector, rev, buffer)` call, over the seeds
    /// Lazily creates the block-span event pair (two CU_EVENT_DEFAULT events
    /// reused across chunks).
    #[cfg(feature = "ref-loc-buckets")]
    fn ensure_block_span_events(&mut self) -> Result<(), DriverError> {
        if self.block_span_start.is_none() {
            self.block_span_start = Some(timed_event(self.stream.context())?);
            self.block_span_end = Some(timed_event(self.stream.context())?);
        }
        Ok(())
    }

    /// One `SeedAndFilter(seed_offset_vector, rev, buffer)` call, over the seeds
    /// already staged in `slot` by [`upload_seeds`](Self::upload_seeds).
    pub fn seed_and_filter(
        &mut self,
        slot: usize,
        rev: bool,
    ) -> Result<FilterOutput, Box<dyn std::error::Error>> {
        self.seed_and_filter_with(slot, rev, false, &mut |_, _| Ok(()))
    }

    /// [`seed_and_filter`](Self::seed_and_filter) that calls `lull` at the
    /// moments of [`Lull`] (each at most once, and not at all on the early
    /// exits: the caller must not depend on it) and, with `defer_dedup`, hands
    /// the HSPs back per chunk without the host sort and dedup.
    pub fn seed_and_filter_with(
        &mut self,
        slot: usize,
        rev: bool,
        defer_dedup: bool,
        lull: &mut LullWork<'_>,
    ) -> Result<FilterOutput, Box<dyn std::error::Error>> {
        if !host_trace::on() {
            return self.seed_and_filter_inner(slot, rev, defer_dedup, lull);
        }
        host_trace::BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut traced = |engine: &mut Engine, at: Lull| {
            host_trace::timed(&host_trace::LULL_NS, || lull(engine, at))
        };
        host_trace::timed(&host_trace::BATCH_NS, || {
            self.seed_and_filter_inner(slot, rev, defer_dedup, &mut traced)
        })
    }

    fn seed_and_filter_inner(
        &mut self,
        slot: usize,
        rev: bool,
        defer_dedup: bool,
        lull: &mut LullWork<'_>,
    ) -> Result<FilterOutput, Box<dyn std::error::Error>> {
        // Refuse to run against a query that was never swapped in.
        //
        // `Engine::new` is reference-only, so a fresh engine starts with
        // no query. Forgetting `swap_query` compiles cleanly — the fields were
        // removed from `EngineConfig`, not renamed, so nothing type-checks their
        // presence — and the whole pipeline would then run against an empty query
        // and emit zero HSPs with no error. That already happened once during the
        // reference-only refactor and only the oracle caught it.
        //
        // An `Err`, not a `debug_assert`: this is an internal invariant that must
        // hold in release builds, which is where a whole-genome run happens.
        if self.query_len == 0 {
            return Err("Engine has no query: swap_query() must be called before \
                 seed_and_filter()"
                .into());
        }
        let num_seeds = self.seed_len[slot];
        // Round 122: a batch is on folded seeds unless the caller said plain.
        let folded = !std::mem::take(&mut self.plain_once) && self.fold_alt.is_some();
        self.use_tables(folded);

        // Allocation and transfer are timed apart, because the
        // remedies are different — a grown-once buffer fixes one and does
        // nothing for the other.
        let t = Instant::now();
        // `uninitialized_async` would be the honest allocation-only probe, but
        // ZLUDA does not implement cuMemAllocAsync (DriverError 801), so the
        // zeroing memset is counted as part of allocation. That is the right
        // grouping for the decision anyway: a grown-once buffer removes both.
        // N7. Both memsets are provably dead: `d_seeds` is fully
        // overwritten by the copy below, and `d_hit_num` by `find_num_hits`'
        // grid-stride loop over every `id < num_seeds`, before `scan_blocks`
        // reads it. Under ZLUDA this cut `alloc seeds + counts` 38.3 -> 31.3 ms
        // but left whole runtime inside noise, so production keeps the zeroing.
        //
        // N3 instead keeps both buffers across batches. The two are
        // deliberately separate experiments: N7 changes initialization, N3
        // changes lifetime.
        if self.persistent_seed_buffers {
            // SAFETY: contents are fully overwritten before any read — see the
            // N7 note above. Spare capacity is never touched because every
            // kernel and copy is bounded by `num_seeds`.
            unsafe {
                reserve(
                    &mut self.buf_hit_num,
                    &self.stream,
                    num_seeds as usize,
                    &mut self.pipeline_syncs,
                )?;
            }
        } else {
            // N3's rejected arm. The seed slots stay persistent regardless: an
            // upload may be in flight into one of them, and reallocating under a
            // queued copy is a use-after-free.
            #[cfg(feature = "nvidia-uninit-seed-buffers")]
            // SAFETY: as above.
            {
                self.buf_hit_num =
                    unsafe { uninitialized::<u32>(&self.stream, num_seeds as usize)? };
            }
            #[cfg(not(feature = "nvidia-uninit-seed-buffers"))]
            {
                self.buf_hit_num = DeviceBuffer::<u32>::zeroed(&self.stream, num_seeds as usize)?;
            }
        }
        // The allocation's memset is stream-ordered ahead of every kernel that
        // reads these buffers, so waiting here only priced the memset — that is a
        // stage sync, not a dependency.
        if !self.async_stages {
            self.stream.synchronize()?;
            self.stage_syncs += 1;
        }
        self.phases.add("alloc seeds + counts", t.elapsed());

        // Phase 3: the upload was issued a batch ago on the copy stream, so all
        // that is left is the ordering edge. `query()` never blocks and tells us
        // whether the overlap actually happened — a stall is the copy still in
        // flight when its compute wanted it, which is the honest measure of
        // "overlap > 0".
        if self.async_seed_copy && num_seeds > 0 {
            if !self.copy_ready[slot].query()? {
                self.seed_copy_stalls += 1;
            }
            self.stream.wait(&self.copy_ready[slot])?;
        }

        // Round 124: a folded batch that will be written from its seeds is
        // counted by one kernel from the folded table's rows.
        let fused = folded && self.fused_count_now();
        let stage = self.stage();
        if fused {
            self.launch_fold_seed_counts(slot, num_seeds)?;
        } else if let Some(p) = &self.paired {
            // `--paired`: count only the bucket positions inside the seed's own
            // pair, and keep where that slice starts for `find_hits_dense_paired`.
            // SAFETY: grid-stride kernel; `d_seed_lo` is grown to one slot per
            // seed here and fully written by the kernel before it is read.
            unsafe {
                reserve(
                    &mut self.buf_seed_lo,
                    &self.stream,
                    num_seeds as usize,
                    &mut self.pipeline_syncs,
                )?;
                self.module.find_num_hits_paired(
                    &self.stream,
                    elementwise(),
                    num_seeds,
                    &self.index_table,
                    &self.pos_table,
                    &self.seed_slots[slot],
                    &p.q_start,
                    &p.t_lo,
                    &p.t_hi,
                    &mut self.buf_hit_num,
                    &mut self.buf_seed_lo,
                )?;
            }
        } else {
            // Round 110: the group kernel needs each seed's start index beside
            // its count.
            #[cfg(feature = "find-hits-groups")]
            let with_lo = self.find_hits_groups != FindHitsGroups::Off;
            #[cfg(not(feature = "find-hits-groups"))]
            let with_lo = false;
            #[cfg(feature = "find-hits-groups")]
            if with_lo {
                // SAFETY: as below; `buf_seed_lo` is grown to one slot per seed
                // and fully written before `find_hits_groups` reads it.
                unsafe {
                    reserve(
                        &mut self.buf_seed_lo,
                        &self.stream,
                        num_seeds as usize,
                        &mut self.pipeline_syncs,
                    )?;
                    self.module.find_num_hits_lo(
                        &self.stream,
                        elementwise(),
                        num_seeds,
                        &self.index_table,
                        &self.seed_slots[slot],
                        &mut self.buf_hit_num,
                        &mut self.buf_seed_lo,
                    )?;
                }
            }
            if !with_lo {
                // SAFETY: grid-stride kernel; `index_table` covers every seed value
                // and `d_hit_num` has one slot per seed.
                unsafe {
                    self.module.find_num_hits(
                        &self.stream,
                        elementwise(),
                        num_seeds,
                        &self.index_table,
                        &self.seed_slots[slot],
                        &mut self.buf_hit_num,
                    )?;
                }
            }
        }
        self.end_stage(
            stage,
            if fused {
                "fold_seed_counts"
            } else {
                "find_num_hits"
            },
        )?;

        let hit_counts = if self.collect_hit_stats || self.census.is_some() {
            // Diagnostic-only: raw counts are otherwise never needed on the
            // host. Copy the initialized prefix once, before the in-place scan.
            let t = Instant::now();
            let raw = copy_prefix(&self.buf_hit_num, &self.stream, num_seeds as usize)?;
            self.absorb_sync()?;
            if let Some(c) = self.census.as_mut() {
                c.observe_counts(&raw);
            }
            self.phases.add("hit-distribution stats", t.elapsed());
            Some(raw)
        } else {
            None
        };

        // The cumulative counts stay on the device. A block-local
        // scan plus an add-back turns them into the global inclusive scan, and
        // only the per-block sums (num_seeds/256 u32) cross the bus instead of
        // the whole array in each direction.
        // Round 124: a fused batch has no count per seed. What is scanned is its
        // runs' counts per bucket, which the group kernel needs scanned anyway,
        // and their block sums give the batch's hit total (saturating: see
        // `scan_blocks_sat`).
        let scan_len = if fused {
            self.fused_scan_len(num_seeds)
        } else {
            num_seeds
        };
        let blocks = scan_len.div_ceil(SCAN_BLOCK);
        // SAFETY: `scan_blocks` writes one sum for every launched block.
        unsafe {
            reserve(
                &mut self.scr_hit_sums,
                &self.stream,
                blocks as usize,
                &mut self.pipeline_syncs,
            )?;
        }
        let scan_cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (SCAN_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stage = self.stage();
        if fused {
            self.launch_scan_sat(scan_cfg, scan_len)?;
        } else {
            // SAFETY: one element per thread over `num_seeds`, one sum slot per
            // block.
            unsafe {
                self.module.scan_blocks(
                    &self.stream,
                    scan_cfg,
                    &mut self.buf_hit_num,
                    &mut self.scr_hit_sums,
                    num_seeds,
                )?;
            }
        }
        self.end_stage(stage, "scan_blocks")?;

        let t = Instant::now();
        lull(self, Lull::Counts)?;
        self.phases.add("host work under kernels", t.elapsed());

        let t = Instant::now();
        // REQUIRED: the host computes the exclusive scan of the
        // block sums and the total decides the chunk walk, so this round trip is a
        // real dependency. Moving the reduction onto the device is a separate
        // change; this path does not redesign it.
        let mut sums = copy_prefix(&self.scr_hit_sums, &self.stream, blocks as usize)?;
        self.absorb_sync()?;
        // `--paired`: a call holding several pairs is only exact while it stays
        // one MAX_HITS sub-chunk (dedup runs per sub-chunk, and a pair's
        // standalone run has its own walk). Past that, stop here — the stream
        // is drained and no slot, pinned or autotune state has been touched,
        // like the `num_hits == 0` return below — and let the driver re-issue
        // one pair per call, which is exactly the standalone call.
        if self.paired.is_some() && self.paired_multi {
            let total: u64 = sums.iter().map(|&s| u64::from(s)).sum();
            if total > u64::from(self.max_hits) {
                self.phases.add("block-sum round trip", t.elapsed());
                return Ok(FilterOutput {
                    hsps: Vec::new(),
                    num_hits: 0,
                    raw_hsps: 0,
                    raw: Vec::new(),
                    audit: Vec::new(),
                    oversized: true,
                    chunks: Vec::new(),
                });
            }
        }
        // Round 122: a folded batch that the plain order would cut into chunks
        // (the walk runs once per `max_hits` hits) is not this path's: the cuts
        // fall at plain seeds. As above, nothing has been touched yet.
        if folded {
            let total: u64 = sums.iter().map(|&s| u64::from(s)).sum();
            self.fold_seen += 1;
            if total >= u64::from(self.max_hits) {
                fold_totals::REISSUED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.fold_again += 1;
                // A batch sent again costs a count pass and its seeds
                // generated on the device, about what the fold saves on a
                // batch. Where most batches are cut (a small `--max-hits`)
                // the fold only costs.
                if self.fold_again >= FOLD_AGAIN_MIN && self.fold_again > self.fold_seen / 4 {
                    self.release_fold("a quarter of its batches are cut at --max-hits");
                    // The cap and the pair are the run's: the next reference
                    // would build its fold, try 16 batches twice and drop it.
                    fold_totals::CUT.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                self.phases.add("block-sum round trip", t.elapsed());
                return Ok(FilterOutput {
                    hsps: Vec::new(),
                    num_hits: 0,
                    raw_hsps: 0,
                    raw: Vec::new(),
                    audit: Vec::new(),
                    oversized: true,
                    chunks: Vec::new(),
                });
            }
            fold_totals::BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let num_hits = exclusive_scan_hit_totals(&mut sums)?;
        // SAFETY: queued before `add_block_offsets`, which reads it; `sums`
        // lives to the end of this call, past every wait on the stream.
        unsafe {
            upload(
                &mut self.scr_hit_offsets,
                &self.stream,
                &sums,
                &mut self.pipeline_syncs,
            )?;
        }
        self.phases.add("block-sum round trip", t.elapsed());

        let stage = self.stage();
        if fused {
            self.launch_add_run_offsets(scan_cfg, scan_len)?;
        } else {
            // SAFETY: same geometry as `scan_blocks`, so block `b` reads offset
            // `b`.
            unsafe {
                self.module.add_block_offsets(
                    &self.stream,
                    scan_cfg,
                    &mut self.buf_hit_num,
                    &self.scr_hit_offsets,
                    num_seeds,
                )?;
            }
        }
        self.end_stage(stage, "add_block_offsets")?;

        let mut out = FilterOutput {
            hsps: Vec::new(),
            num_hits,
            raw_hsps: 0,
            raw: Vec::new(),
            audit: Vec::new(),
            oversized: false,
            chunks: Vec::new(),
        };
        // The gate's lull is offered once per call, on its first chunk.
        #[cfg_attr(not(feature = "dense-anchors"), allow(unused_variables, unused_mut))]
        let mut lulled_gate = false;
        if num_hits == 0 {
            if self.collect_hit_stats {
                let t = Instant::now();
                self.hit_stats
                    .observe_launch(hit_counts.as_deref().expect("hit stats requested counts"));
                self.phases.add("hit-distribution stats", t.elapsed());
            }
            // The scratch buffers are the engine's now, but keep the wait: this is
            // the rare exit, and it used to be needed because
            // `d_block_sums`/`d_offsets` were freed on the way out and
            // `add_block_offsets` may still be queued reading them.
            self.sync_pipeline()?;
            return Ok(out);
        }

        let t = Instant::now();
        let chunks = if num_hits <= self.max_hits {
            // The overwhelmingly common case: one chunk, so the walk needs only
            // the total, which the block-sum scan already produced.
            vec![(0, num_seeds - 1, 0, num_hits)]
        } else {
            // Rare: the cap actually splits this call, and the exact KegAlign
            // walk needs element granularity. Pay for the array only here.
            if let Some(counts) = hit_counts.as_ref() {
                let mut total = 0u32;
                let cumulative: Vec<u32> = counts
                    .iter()
                    .map(|&count| {
                        total += count;
                        total
                    })
                    .collect();
                chunk_limits(&cumulative, self.max_hits, self.hit_capacity).map_err(|e| {
                    format!(
                        "{e}; seed_batch={{slot={slot} num_seeds={num_seeds} num_hits={num_hits} rev={rev} cap={} capacity={}}}",
                        self.max_hits, self.hit_capacity
                    )
                })?
            } else if self.chunk_walk_full {
                // `HSPZ_CHUNK_WALK=full`: the pre-round-85 full-copy path.
                let cumulative = copy_prefix(&self.buf_hit_num, &self.stream, num_seeds as usize)?;
                self.absorb_sync()?;
                chunk_limits(&cumulative, self.max_hits, self.hit_capacity).map_err(|e| {
                    format!(
                        "{e}; seed_batch={{slot={slot} num_seeds={num_seeds} num_hits={num_hits} rev={rev} cap={} capacity={}}}",
                        self.max_hits, self.hit_capacity
                    )
                })?
            } else {
                // Round 85 (default): the same walk through `sums` plus one
                // <=256-element D2H per boundary block, instead of the whole
                // cumulative array. Same stream, so every fetch is ordered
                // after `add_block_offsets`; same boundaries, same errors.
                let buf = &self.buf_hit_num;
                let stream = &self.stream;
                let scratch = &mut self.chunk_block_scratch;
                let scan = SCAN_BLOCK as usize;
                let mut fetch = |b: usize| -> Result<Vec<u32>, String> {
                    let first = b * scan;
                    let len = (num_seeds as usize - first).min(scan);
                    copy_range_into(buf, stream, first, len, scratch)
                        .map_err(|e| format!("sparse chunk-walk block fetch failed: {e}"))?;
                    // The DMA target above is the reused Engine buffer; the
                    // 1 KiB host clone is noise against the ~6.8 MiB copy it
                    // replaces.
                    Ok(scratch.clone())
                };
                let chunks = chunk_limits_sparse(
                    &mut fetch,
                    &sums,
                    num_seeds,
                    num_hits,
                    self.max_hits,
                    self.hit_capacity,
                )
                .map_err(|e| {
                    format!(
                        "{e}; seed_batch={{slot={slot} num_seeds={num_seeds} num_hits={num_hits} rev={rev} cap={} capacity={}}}",
                        self.max_hits, self.hit_capacity
                    )
                })?;
                self.absorb_sync()?;
                chunks
            }
        };
        self.phases.add("chunk prep (lower_bound)", t.elapsed());

        if self.collect_hit_stats {
            let t = Instant::now();
            let counts = hit_counts.as_deref().expect("hit stats requested counts");
            for &(start, limit, _, _) in &chunks {
                self.hit_stats
                    .observe_launch(&counts[start as usize..=limit as usize]);
            }
            self.phases.add("hit-distribution stats", t.elapsed());
        }

        for (start_seed_index, limit_pos, start_hit_val, end_hit_val) in chunks.iter().copied() {
            let iter_num_seeds = limit_pos + 1 - start_seed_index;
            let iter_num_hits = end_hit_val - start_hit_val;
            if iter_num_hits == 0 {
                continue;
            }
            // Round 87 (blocks): every chunk runs its block's production
            // path; timed chunks feed the block accounting at the
            // survivor-total sync below. Warm chunks and Forced/Decided
            // chunks never time.
            // `gate-sieve`: this unit's class constants when the bound is on.
            // The guard samples the candidate fraction on every `period`th
            // chunk and holds the plain gate while the bound rejects too little.
            #[cfg(feature = "gate-sieve")]
            let (sieve_args, sieve_sample) = match (self.sieve.args, self.sieve.guard.next_batch())
            {
                (None, _) => (None, false),
                (Some(_), GuardStep::Plain) => (None, false),
                (args, GuardStep::Sieve) => (args, false),
                (args, GuardStep::Sample) => (args, true),
            };
            #[cfg(not(feature = "gate-sieve"))]
            #[cfg_attr(not(feature = "dense-anchors"), allow(unused_variables))]
            let sieve_args: Option<(i32, i32, i32, i32)> = None;
            #[cfg(feature = "ref-loc-buckets")]
            let (bucket_pass, time_span) = if self.bucket_planes && sieve_args.is_none() {
                // A plane-sized window does nothing for the plain gate.
                (false, false)
            } else {
                let max_hits = self.max_hits;
                let timed = self.bucket_mode.auto_chunk(iter_num_hits, max_hits);
                if timed {
                    self.ensure_block_span_events()?;
                }
                (self.bucket_mode.production_path(), timed)
            };

            // Round 116: survivors by sorted slot (see `survivor_slots_request`).
            #[cfg(all(feature = "ref-loc-buckets", feature = "gate-sieve"))]
            let slot_order = bucket_pass
                && self.survivor_slots
                && self.bucket_match
                && !self.dump_raw
                && self.census.is_none()
                && order_check_limit() == 0;
            #[cfg(not(all(feature = "ref-loc-buckets", feature = "gate-sieve")))]
            let slot_order = false;
            // Round 119: the bucket order written from the seeds, in
            // which case no raw anchor is written at all.
            #[cfg(all(
                feature = "find-hits-groups",
                feature = "ref-loc-buckets",
                feature = "gate-sieve"
            ))]
            let seeds_can =
                slot_order && self.paired.is_none() && !self.buf_bucket_nibbles.is_empty();
            #[cfg(all(
                feature = "find-hits-groups",
                feature = "ref-loc-buckets",
                feature = "gate-sieve"
            ))]
            let from_seeds = seeds_can && self.hits_bucketed == HitsBucketed::On;
            #[cfg(not(all(
                feature = "find-hits-groups",
                feature = "ref-loc-buckets",
                feature = "gate-sieve"
            )))]
            let from_seeds = false;
            if fused && !from_seeds {
                return Err(
                    "internal: a batch counted from the folded rows must be written from \
                            its seeds"
                        .into(),
                );
            }

            // Round 122: the folded tables were paid for with capacity. A chunk
            // above what is left (only a batch cut into chunks has one, and
            // those are on the plain tables) takes the memory back.
            if !folded && self.fold_alt.is_some() && iter_num_hits > self.fold_capacity {
                self.release_fold("a chunk needs its memory");
            }

            let t = Instant::now();
            #[cfg(not(feature = "dense-anchors"))]
            // SAFETY: `find_hits` and `find_hsps` overwrite every element read
            // by the control path.
            unsafe {
                reserve(
                    &mut self.buf_hsp,
                    &self.stream,
                    iter_num_hits as usize,
                    &mut self.pipeline_syncs,
                )?;
                reserve(
                    &mut self.buf_done,
                    &self.stream,
                    iter_num_hits as usize,
                    &mut self.pipeline_syncs,
                )?;
            }
            #[cfg(feature = "dense-anchors")]
            // SAFETY: `find_hits_dense` and `mark_score_survivors` overwrite
            // every active anchor and flag before either is read.
            unsafe {
                // No raw anchor is written when the chunk's bucket order comes
                // from its seeds.
                if !from_seeds {
                    reserve(
                        &mut self.buf_anchor,
                        &self.stream,
                        iter_num_hits as usize,
                        &mut self.pipeline_syncs,
                    )?;
                }
                // One byte per hit, for the gate that writes bytes: the bucket
                // pass and the sieve write keep bits instead.
                if !bucket_pass && sieve_args.is_none() {
                    reserve(
                        &mut self.buf_flags,
                        &self.stream,
                        iter_num_hits as usize,
                        &mut self.pipeline_syncs,
                    )?;
                }
                // One candidate word per 32 hits, written by the batches the
                // sieve's guard samples and under its count diagnostic.
                #[cfg(feature = "gate-sieve")]
                if self.sieve.count || self.sieve.guard.period > 0 {
                    reserve(
                        &mut self.sieve.cand_bits,
                        &self.stream,
                        iter_num_hits.div_ceil(32) as usize,
                        &mut self.pipeline_syncs,
                    )?;
                }
            }
            self.phases.add("alloc hit buffers", t.elapsed());

            // A chunk written from its seeds launches nothing here, and has no
            // stage (it was timed as `find_hits (none)` until 0.1.2: 0.4 s of
            // "GPU time" per genome that no kernel ran).
            let stage = if from_seeds { None } else { Some(self.stage()) };
            // `--paired` always takes the thread kernel: the output does not
            // depend on the choice, and pair densities sit far below the warp
            // cutoff anyway.
            let warp_find_hits =
                self.paired.is_none() && use_warp_find_hits(iter_num_hits, iter_num_seeds);
            // Round 110: the same anchors from runs of 32 seeds per warp.
            #[cfg(feature = "find-hits-groups")]
            let groups = if self.paired.is_none() {
                self.find_hits_groups
            } else {
                FindHitsGroups::Off
            };
            #[cfg(feature = "find-hits-groups")]
            let by_groups = groups == FindHitsGroups::On;
            #[cfg(not(feature = "find-hits-groups"))]
            let by_groups = false;
            if !from_seeds {
                #[cfg(feature = "find-hits-groups")]
                if by_groups {
                    use std::sync::atomic::Ordering::Relaxed;
                    find_hits_groups_totals::CHUNKS.fetch_add(1, Relaxed);
                    find_hits_groups_totals::HITS.fetch_add(u64::from(iter_num_hits), Relaxed);
                    self.launch_find_hits_groups(
                        slot,
                        start_seed_index,
                        start_hit_val,
                        iter_num_seeds,
                    )?;
                }
                // SAFETY: one thread or warp per seed; every store lands inside
                // `0..iter_num_hits` by construction of the prefix offsets.
                #[cfg(not(feature = "dense-anchors"))]
                unsafe {
                    self.module.find_hits(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: (iter_num_seeds.div_ceil(BLOCK_SIZE), 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &self.index_table,
                        &self.pos_table,
                        &self.seed_slots[slot],
                        self.seed_size,
                        &self.buf_hit_num,
                        &mut self.buf_hsp,
                        start_seed_index,
                        start_hit_val,
                        iter_num_seeds,
                    )?;
                }
                #[cfg(feature = "dense-anchors")]
                if self.paired.is_some() {
                    // SAFETY: as `find_hits_dense`; each seed walks only the
                    // `[seed_lo, seed_lo + count)` slice `find_num_hits_paired`
                    // counted, so the prefix offsets bound every store.
                    unsafe {
                        self.module.find_hits_dense_paired(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (iter_num_seeds.div_ceil(BLOCK_SIZE), 1, 1),
                                block_dim: (BLOCK_SIZE, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &self.pos_table,
                            &self.buf_seed_lo,
                            &self.seed_slots[slot],
                            self.seed_size,
                            &self.buf_hit_num,
                            &mut self.buf_anchor,
                            start_seed_index,
                            start_hit_val,
                            iter_num_seeds,
                        )?;
                    }
                }
                #[cfg(feature = "dense-anchors")]
                if self.paired.is_none() && !by_groups {
                    unsafe {
                        #[cfg(feature = "find-hits-warp")]
                        if warp_find_hits {
                            self.module.find_hits_dense_warp(
                                &self.stream,
                                LaunchConfig {
                                    grid_dim: (iter_num_seeds.div_ceil(NUM_WARPS as u32), 1, 1),
                                    block_dim: (BLOCK_SIZE, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                &self.index_table,
                                &self.pos_table,
                                &self.seed_slots[slot],
                                self.seed_size,
                                &self.buf_hit_num,
                                &mut self.buf_anchor,
                                start_seed_index,
                                start_hit_val,
                                iter_num_seeds,
                            )?;
                        } else {
                            self.module.find_hits_dense(
                                &self.stream,
                                LaunchConfig {
                                    grid_dim: (iter_num_seeds.div_ceil(BLOCK_SIZE), 1, 1),
                                    block_dim: (BLOCK_SIZE, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                &self.index_table,
                                &self.pos_table,
                                &self.seed_slots[slot],
                                self.seed_size,
                                &self.buf_hit_num,
                                &mut self.buf_anchor,
                                start_seed_index,
                                start_hit_val,
                                iter_num_seeds,
                            )?;
                        }
                        #[cfg(not(feature = "find-hits-warp"))]
                        self.module.find_hits_dense(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (iter_num_seeds.div_ceil(BLOCK_SIZE), 1, 1),
                                block_dim: (BLOCK_SIZE, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &self.index_table,
                            &self.pos_table,
                            &self.seed_slots[slot],
                            self.seed_size,
                            &self.buf_hit_num,
                            &mut self.buf_anchor,
                            start_seed_index,
                            start_hit_val,
                            iter_num_seeds,
                        )?;
                    }
                }
            }
            if let Some(stage) = stage {
                self.end_stage(
                    stage,
                    if by_groups {
                        "find_hits (groups)"
                    } else if warp_find_hits {
                        "find_hits (warp)"
                    } else {
                        "find_hits"
                    },
                )?;
            }
            #[cfg(feature = "find-hits-groups")]
            if groups == FindHitsGroups::Check {
                self.check_find_hits_groups(
                    slot,
                    start_seed_index,
                    start_hit_val,
                    iter_num_seeds,
                    iter_num_hits,
                )?;
            }
            // Round 2: the compiled-in pass is a runtime choice, so one binary
            // covers both a card the window helps and one it does not
            // (`HSPZ_REF_BUCKETS=0`). Round 87 `auto` runs alternating LONG
            // blocks of each production path and commits to the faster
            // settled ns/hit.
            #[cfg(not(feature = "ref-loc-buckets"))]
            let bucket_pass = false;
            // Number of bitmask words the gate writes and the survivor scan
            // reads: one per 32 sorted slots.
            #[cfg(feature = "ref-loc-buckets")]
            let flag_words = iter_num_hits.div_ceil(32);

            // Round 83: stable counting sort of this chunk's anchors by
            // reference-address bucket, so consecutive warps of the gate gather
            // from one L2-resident reference window. `buf_anchor` is left
            // untouched — `find_hsps` still reads it by survivor id.
            #[cfg(feature = "ref-loc-buckets")]
            if bucket_pass {
                let cblocks = iter_num_hits.div_ceil(SCAN_BLOCK);
                let counts_len = self.n_buckets * cblocks;

                let t = Instant::now();
                // SAFETY: `bucket_count` writes every count and `bucket_scatter`
                // every one of the `iter_num_hits` permuted slots before the
                // gate reads them.
                unsafe {
                    reserve(
                        &mut self.buf_sorted_anchor,
                        &self.stream,
                        iter_num_hits as usize,
                        &mut self.pipeline_syncs,
                    )?;
                    if !slot_order {
                        reserve(
                            &mut self.buf_sorted_idx,
                            &self.stream,
                            iter_num_hits as usize,
                            &mut self.pipeline_syncs,
                        )?;
                    }
                    reserve(
                        &mut self.buf_bucket_counts,
                        &self.stream,
                        counts_len as usize,
                        &mut self.pipeline_syncs,
                    )?;
                    reserve(
                        &mut self.buf_flags_bits,
                        &self.stream,
                        flag_words as usize,
                        &mut self.pipeline_syncs,
                    )?;
                }
                self.phases.add("alloc bucket buffers", t.elapsed());

                // Round 87 (blocks): the ON span starts after the
                // reserves, so host allocation stays outside the event
                // pair; it covers the bucket pass plus the reordered
                // gate and count below.
                #[cfg(feature = "ref-loc-buckets")]
                if time_span {
                    self.block_span_start
                        .as_ref()
                        .expect("block span events ensured above")
                        .record(&self.stream)?;
                }

                // Round 119: the bucket order from the seeds in
                // place of the count and the scatter, or beside them.
                #[cfg(all(
                    feature = "find-hits-groups",
                    feature = "ref-loc-buckets",
                    feature = "gate-sieve"
                ))]
                if from_seeds {
                    use std::sync::atomic::Ordering::Relaxed;
                    hits_bucketed_totals::CHUNKS.fetch_add(1, Relaxed);
                    hits_bucketed_totals::HITS.fetch_add(u64::from(iter_num_hits), Relaxed);
                    self.launch_hits_bucketed(slot, start_seed_index, iter_num_seeds, fused)?;
                }
                if !from_seeds {
                    let stage = self.stage();
                    // SAFETY: one hit per thread; thread `v < n_buckets` of block
                    // `bid` owns exactly slot `v * cblocks + bid`.
                    unsafe {
                        let config = LaunchConfig {
                            grid_dim: (cblocks, 1, 1),
                            block_dim: (SCAN_BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        if self.bucket_match {
                            self.module.bucket_count_match(
                                &self.stream,
                                config,
                                &self.buf_anchor,
                                iter_num_hits,
                                self.bucket_shift,
                                cblocks,
                                self.n_buckets,
                                &mut self.buf_bucket_counts,
                            )?;
                        } else {
                            self.module.bucket_count(
                                &self.stream,
                                config,
                                &self.buf_anchor,
                                iter_num_hits,
                                self.bucket_shift,
                                cblocks,
                                self.n_buckets,
                                &mut self.buf_bucket_counts,
                            )?;
                        }
                    }
                    self.end_stage(stage, "ref_bucket_count")?;

                    let sblocks = counts_len.div_ceil(SCAN_BLOCK);
                    // SAFETY: `scan_blocks` writes one sum for every
                    // launched block.
                    unsafe {
                        reserve(
                            &mut self.scr_run_sums,
                            &self.stream,
                            sblocks as usize,
                            &mut self.pipeline_syncs,
                        )?;
                    }
                    let stage = self.stage();
                    // SAFETY: one count per thread, one sum slot per launched block.
                    unsafe {
                        self.module.scan_blocks(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (sblocks, 1, 1),
                                block_dim: (SCAN_BLOCK, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &mut self.buf_bucket_counts,
                            &mut self.scr_run_sums,
                            counts_len,
                        )?;
                    }
                    self.end_stage(stage, "ref_bucket_scan")?;

                    // Round 2: the upper level of the scan runs in ONE block on the
                    // device instead of crossing the bus twice per chunk. `sblocks`
                    // is ~6.6k block sums today (~27k at a 64M cap); the kernel
                    // loops over any length, and the totals it adds are hit counts,
                    // so `u32` is exact.
                    let stage = self.stage();
                    // SAFETY: one block, in place over the `sblocks` block sums.
                    unsafe {
                        self.module.scan_exclusive_one_block(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (1, 1, 1),
                                block_dim: (kernels::ONE_BLOCK_THREADS, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &mut self.scr_run_sums,
                            sblocks,
                        )?;
                    }
                    self.end_stage(stage, "ref_bucket_scan")?;

                    let stage = self.stage();
                    // SAFETY: same geometry as `scan_blocks`, so block `b` reads
                    // offset `b`.
                    unsafe {
                        self.module.add_block_offsets(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (sblocks, 1, 1),
                                block_dim: (SCAN_BLOCK, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &mut self.buf_bucket_counts,
                            &self.scr_run_sums,
                            counts_len,
                        )?;
                    }
                    self.end_stage(stage, "ref_bucket_scan")?;

                    let stage = self.stage();
                    // SAFETY: same geometry as `bucket_count`, and the inclusive
                    // prefix partitions `0..iter_num_hits` into one slot per hit.
                    unsafe {
                        let config = LaunchConfig {
                            grid_dim: (cblocks, 1, 1),
                            block_dim: (SCAN_BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        #[cfg(feature = "gate-sieve")]
                        if slot_order {
                            self.module.bucket_scatter_match_anchors(
                                &self.stream,
                                config,
                                &self.buf_anchor,
                                iter_num_hits,
                                self.bucket_shift,
                                cblocks,
                                self.n_buckets,
                                &self.buf_bucket_counts,
                                &mut self.buf_sorted_anchor,
                            )?;
                        }
                        if slot_order {
                            // Launched above.
                        } else if self.bucket_match {
                            self.module.bucket_scatter_match(
                                &self.stream,
                                config,
                                &self.buf_anchor,
                                iter_num_hits,
                                self.bucket_shift,
                                cblocks,
                                self.n_buckets,
                                &self.buf_bucket_counts,
                                &mut self.buf_sorted_anchor,
                                &mut self.buf_sorted_idx,
                            )?;
                        } else {
                            self.module.bucket_scatter(
                                &self.stream,
                                config,
                                &self.buf_anchor,
                                iter_num_hits,
                                self.bucket_shift,
                                cblocks,
                                self.n_buckets,
                                &self.buf_bucket_counts,
                                &mut self.buf_sorted_anchor,
                                &mut self.buf_sorted_idx,
                            )?;
                        }
                    }
                    self.end_stage(stage, "ref_bucket_scatter")?;
                }
                #[cfg(all(
                    feature = "find-hits-groups",
                    feature = "ref-loc-buckets",
                    feature = "gate-sieve"
                ))]
                if let HitsBucketed::Check(limit) = self.hits_bucketed {
                    use std::sync::atomic::Ordering::Relaxed;
                    if seeds_can
                        && hits_bucketed_totals::CHECKED_CHUNKS.load(Relaxed) < u64::from(limit)
                    {
                        self.check_hits_bucketed(
                            slot,
                            start_seed_index,
                            iter_num_seeds,
                            iter_num_hits,
                        )?;
                    }
                }
                #[cfg(feature = "ref-loc-buckets")]
                if order_check_limit() > 0 {
                    static BUCKET_CHECK_DONE: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let check_k = order_check_limit();
                    let check_c =
                        BUCKET_CHECK_DONE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if check_c < check_k {
                        self.stream.synchronize()?;
                        let check_n = iter_num_hits as usize;
                        let check_sorted_anchor =
                            copy_prefix(&self.buf_sorted_anchor, &self.stream, check_n)?;
                        let check_sorted_idx =
                            copy_prefix(&self.buf_sorted_idx, &self.stream, check_n)?;
                        let check_orig_anchor =
                            copy_prefix(&self.buf_anchor, &self.stream, check_n)?;
                        let bucket_of = |a: u64| -> u32 { (a as u32) >> self.bucket_shift };
                        let mut perm_ok = true;
                        let mut perm_detail = String::from("none");
                        {
                            let mut seen = vec![false; check_n];
                            for (p, &idx) in check_sorted_idx.iter().enumerate() {
                                let i = idx as usize;
                                if idx as usize >= check_n || seen[i] {
                                    perm_ok = false;
                                    perm_detail = format!("p={p} idx={idx} n={check_n}");
                                    break;
                                }
                                seen[i] = true;
                            }
                        }
                        let mut monotone_ok = true;
                        let mut monotone_detail = String::from("none");
                        let mut prev_b = 0u32;
                        for (p, &a) in check_sorted_anchor.iter().enumerate() {
                            let b = bucket_of(a);
                            if p > 0 && b < prev_b {
                                monotone_ok = false;
                                monotone_detail =
                                    format!("p={p} bucket={b} prev={prev_b} anchor={a:#x}");
                                break;
                            }
                            prev_b = b;
                        }
                        let mut stable_ok = true;
                        let mut stable_detail = String::from("none");
                        for p in 1..check_n {
                            if bucket_of(check_sorted_anchor[p])
                                == bucket_of(check_sorted_anchor[p - 1])
                                && check_sorted_idx[p] <= check_sorted_idx[p - 1]
                            {
                                stable_ok = false;
                                stable_detail = format!(
                                    "p={p} idx={} prev_idx={}",
                                    check_sorted_idx[p],
                                    check_sorted_idx[p - 1]
                                );
                                break;
                            }
                        }
                        let mut anchors_ok = true;
                        let mut anchors_detail = String::from("none");
                        for (p, &idx) in check_sorted_idx.iter().enumerate() {
                            let i = idx as usize;
                            if i >= check_n || check_orig_anchor[i] != check_sorted_anchor[p] {
                                anchors_ok = false;
                                anchors_detail = format!("p={p} idx={idx}");
                                break;
                            }
                        }
                        let ok_str = |ok: bool| if ok { "ok" } else { "FAIL" };
                        if !perm_ok {
                            eprintln!("#bucket-check-violation chunk={check_c} perm {perm_detail}");
                        }
                        if !monotone_ok {
                            eprintln!(
                                "#bucket-check-violation chunk={check_c} monotone {monotone_detail}"
                            );
                        }
                        if !stable_ok {
                            eprintln!(
                                "#bucket-check-violation chunk={check_c} stable {stable_detail}"
                            );
                        }
                        if !anchors_ok {
                            eprintln!(
                                "#bucket-check-violation chunk={check_c} anchors {anchors_detail}"
                            );
                        }
                        let n_buckets = self.n_buckets as usize;
                        let mut hist = vec![0u32; n_buckets];
                        for &a in &check_sorted_anchor {
                            let b = bucket_of(a) as usize;
                            if b < n_buckets {
                                hist[b] += 1;
                            }
                        }
                        let mut hist_sorted = hist.clone();
                        hist_sorted.sort_unstable();
                        let (hb_min, hb_med, hb_max) = (
                            hist_sorted[0],
                            hist_sorted[n_buckets / 2],
                            hist_sorted[n_buckets - 1],
                        );
                        let ref_len = self.ref_len;
                        let push_sectors = |r: u32, out: &mut Vec<u32>| {
                            let start = r.saturating_sub(64);
                            let mut end = r.saturating_add(32);
                            if end > ref_len {
                                end = ref_len;
                            }
                            if start >= end {
                                return;
                            }
                            for s in (start >> 5)..=((end - 1) >> 5) {
                                out.push(s);
                            }
                        };
                        let mut sectors: Vec<u32> = Vec::new();
                        for &a in &check_sorted_anchor {
                            push_sectors(a as u32, &mut sectors);
                        }
                        sectors.sort_unstable();
                        let mut distinct: u64 = 0;
                        let mut prev_s: Option<u32> = None;
                        for &s in &sectors {
                            if prev_s != Some(s) {
                                distinct += 1;
                                prev_s = Some(s);
                            }
                        }
                        let touches = sectors.len() as u64;
                        let tps = if distinct > 0 {
                            touches as f64 / distinct as f64
                        } else {
                            0.0
                        };
                        let mut bucket_tps_max = 0.0f64;
                        let mut lo = 0usize;
                        while lo < check_n {
                            let b = bucket_of(check_sorted_anchor[lo]);
                            let mut hi = lo + 1;
                            while hi < check_n && bucket_of(check_sorted_anchor[hi]) == b {
                                hi += 1;
                            }
                            let mut v: Vec<u32> = Vec::new();
                            for &a in &check_sorted_anchor[lo..hi] {
                                push_sectors(a as u32, &mut v);
                            }
                            v.sort_unstable();
                            let mut d: u64 = 0;
                            let mut pv: Option<u32> = None;
                            for &s in &v {
                                if pv != Some(s) {
                                    d += 1;
                                    pv = Some(s);
                                }
                            }
                            if d > 0 {
                                bucket_tps_max = bucket_tps_max.max(v.len() as f64 / d as f64);
                            }
                            lo = hi;
                        }
                        eprintln!(
                            "#bucket-check chunk={check_c} n={check_n} perm={} monotone={} stable={} anchors={} B={} shift={} hits_per_bucket={hb_min}/{hb_med}/{hb_max} distinct_sectors_total={distinct} touches_total={touches} touches_per_sector={tps:.2} bucket_tps_max={bucket_tps_max:.2}",
                            ok_str(perm_ok),
                            ok_str(monotone_ok),
                            ok_str(stable_ok),
                            ok_str(anchors_ok),
                            self.n_buckets,
                            self.bucket_shift,
                        );
                        // Round 116: the scatter that writes no raw id, run
                        // beside the one just checked into a buffer of its
                        // own, must write the same anchors word for word.
                        #[cfg(feature = "gate-sieve")]
                        if self.survivor_slots && self.bucket_match {
                            // SAFETY: the kernel writes every one of the
                            // `iter_num_hits` slots before the copy below.
                            let mut d_plain = unsafe {
                                uninitialized::<u64>(&self.stream, iter_num_hits as usize)?
                            };
                            unsafe {
                                self.module.bucket_scatter_match_anchors(
                                    &self.stream,
                                    LaunchConfig {
                                        grid_dim: (cblocks, 1, 1),
                                        block_dim: (SCAN_BLOCK, 1, 1),
                                        shared_mem_bytes: 0,
                                    },
                                    &self.buf_anchor,
                                    iter_num_hits,
                                    self.bucket_shift,
                                    cblocks,
                                    self.n_buckets,
                                    &self.buf_bucket_counts,
                                    &mut d_plain,
                                )?;
                            }
                            let plain = copy_prefix(&d_plain, &self.stream, check_n)?;
                            let differ = plain
                                .iter()
                                .zip(&check_sorted_anchor)
                                .position(|(a, b)| a != b);
                            if let Some(at) = differ {
                                eprintln!(
                                    "#bucket-check-violation chunk={check_c} anchors_only slot={at}"
                                );
                            }
                            eprintln!(
                                "#bucket-check-anchors-only chunk={check_c} n={check_n} equal={}",
                                ok_str(differ.is_none()),
                            );
                        }
                    }
                }
            }

            #[cfg(feature = "dense-anchors")]
            let (materializer_hits, _survivor_offsets) = {
                let query = if rev {
                    &self.query_rc_seq
                } else {
                    &self.query_seq
                };
                // Round 87 (blocks): the OFF span covers the default gate
                // plus the count below. (The ON span started before the
                // bucket pass.)
                #[cfg(feature = "ref-loc-buckets")]
                if time_span && !bucket_pass {
                    self.block_span_start
                        .as_ref()
                        .expect("block span events ensured above")
                        .record(&self.stream)?;
                }
                let stage = self.stage();
                // With `sieve_args` the launch goes to the sieve kernel of its path.
                // The sieve kernels write keep bits, one word per 32 hits, on both
                // gate paths; the plain gate kernel writes a byte per hit.
                let bits_pass = bucket_pass || sieve_args.is_some();
                #[cfg(feature = "gate-sieve")]
                if let Some((um, ua, ub, ubias)) = sieve_args {
                    let strand = rev as usize;
                    let count = (self.sieve.count || sieve_sample) as u32;
                    let words = self.sieve.words.0 | self.sieve.words.1 << 8;
                    let masked = self.sieve.masked;
                    if !bucket_pass {
                        // SAFETY: the gate overwrites every word of the chunk.
                        unsafe {
                            reserve(
                                &mut self.buf_flags_bits,
                                &self.stream,
                                flag_words as usize,
                                &mut self.pipeline_syncs,
                            )?;
                        }
                    }
                    // One launch, four kernels: plain or bucketed anchors, own or
                    // gathered reference groups.
                    macro_rules! launch_sieve {
                        ($kernel:ident, $anchors:expr, $flags:expr) => {
                            self.module.$kernel(
                                &self.stream,
                                LaunchConfig {
                                    grid_dim: (self.hsp_blocks, 1, 1),
                                    block_dim: (HSP_THREADS, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                &self.ref_seq,
                                query,
                                self.ref_len,
                                self.query_len,
                                &self.sub_mat,
                                self.xdrop,
                                self.gate_thresh,
                                iter_num_hits,
                                $anchors,
                                $flags,
                                &self.sieve.ref_pk,
                                &self.sieve.qry_pk[strand],
                                &self.sieve.qry_bad[strand],
                                um,
                                ua,
                                ub,
                                ubias,
                                count,
                                words,
                                &mut self.sieve.cand_bits,
                            )?
                        };
                    }
                    // SAFETY: as the two launches below; the planes were packed
                    // for this reference and this query strand, and the
                    // candidate words are reserved whenever `count` is set.
                    unsafe {
                        match (bucket_pass, self.sieve.gather.unwrap_or(!bucket_pass)) {
                            (false, false) if !masked => launch_sieve!(
                                mark_score_survivors_sieve_open,
                                &self.buf_anchor,
                                &mut self.buf_flags_bits
                            ),
                            (true, false) if !masked => launch_sieve!(
                                mark_score_survivors_reordered_sieve_open,
                                &self.buf_sorted_anchor,
                                &mut self.buf_flags_bits
                            ),
                            (false, false) => launch_sieve!(
                                mark_score_survivors_sieve,
                                &self.buf_anchor,
                                &mut self.buf_flags_bits
                            ),
                            (false, true) => launch_sieve!(
                                mark_score_survivors_sieve_gather,
                                &self.buf_anchor,
                                &mut self.buf_flags_bits
                            ),
                            (true, false) => launch_sieve!(
                                mark_score_survivors_reordered_sieve,
                                &self.buf_sorted_anchor,
                                &mut self.buf_flags_bits
                            ),
                            (true, true) => launch_sieve!(
                                mark_score_survivors_reordered_sieve_gather,
                                &self.buf_sorted_anchor,
                                &mut self.buf_flags_bits
                            ),
                        }
                    }
                }
                // SAFETY: one warp per raw hit; every active flag is written.
                if !bucket_pass && sieve_args.is_none() {
                    unsafe {
                        self.module.mark_score_survivors(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (self.hsp_blocks, 1, 1),
                                block_dim: (HSP_THREADS, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &self.ref_seq,
                            query,
                            self.ref_len,
                            self.query_len,
                            &self.sub_mat,
                            self.xdrop,
                            self.gate_thresh,
                            iter_num_hits,
                            &self.buf_anchor,
                            &mut self.buf_flags,
                        )?;
                    }
                }
                // SAFETY: as above, over the permuted anchors; the gate writes
                // one keep-bit word per 32 sorted slots and `emit_bits` maps
                // those slots back through `buf_sorted_idx`.
                #[cfg(feature = "ref-loc-buckets")]
                if bucket_pass && sieve_args.is_none() {
                    unsafe {
                        self.module.mark_score_survivors_reordered(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (self.hsp_blocks, 1, 1),
                                block_dim: (HSP_THREADS, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &self.ref_seq,
                            query,
                            self.ref_len,
                            self.query_len,
                            &self.sub_mat,
                            self.xdrop,
                            self.gate_thresh,
                            iter_num_hits,
                            &self.buf_sorted_anchor,
                            &mut self.buf_flags_bits,
                        )?;
                    }
                }
                self.end_stage(stage, "score_gate")?;
                #[cfg(feature = "gate-sieve")]
                if sieve_args.is_some() && (self.sieve.count || sieve_sample) {
                    let candidates = if self.sieve.count {
                        self.sieve_audit(rev, iter_num_hits, bucket_pass)?
                    } else {
                        self.sieve_count_candidates(iter_num_hits)?
                    };
                    if sieve_sample {
                        self.sieve
                            .guard
                            .observe(candidates, u64::from(iter_num_hits));
                    }
                }

                // The bitmask packs 32 hits per word, so its scan launches 32x
                // fewer blocks and its block-sum round trip is 32x smaller.
                #[cfg(feature = "ref-loc-buckets")]
                let scan_len = if bits_pass { flag_words } else { iter_num_hits };
                #[cfg(not(feature = "ref-loc-buckets"))]
                let scan_len = iter_num_hits;
                let blocks = scan_len.div_ceil(SCAN_BLOCK);
                // SAFETY: the count kernel writes one sum for every launched
                // block before the host copy reads it.
                unsafe {
                    reserve(
                        &mut self.scr_surv_sums,
                        &self.stream,
                        blocks as usize,
                        &mut self.pipeline_syncs,
                    )?;
                }
                let stage = self.stage();
                if !bits_pass {
                    unsafe {
                        self.module.count_survivors(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (blocks, 1, 1),
                                block_dim: (SCAN_BLOCK, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &self.buf_flags,
                            &mut self.scr_surv_sums,
                            iter_num_hits,
                        )?;
                    }
                }
                #[cfg(feature = "ref-loc-buckets")]
                if bits_pass {
                    unsafe {
                        self.module.count_bits(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (blocks, 1, 1),
                                block_dim: (SCAN_BLOCK, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &self.buf_flags_bits,
                            &mut self.scr_surv_sums,
                            flag_words,
                        )?;
                    }
                }
                self.end_stage(stage, "count_survivors")?;
                // Round 87 (blocks): the span ends after its count;
                // emit/sort run untimed. Covered by the survivor-total sync
                // below, so the zero-survivor early exit commits with no new
                // sync.
                #[cfg(feature = "ref-loc-buckets")]
                if time_span {
                    self.block_span_end
                        .as_ref()
                        .expect("block span events ensured above")
                        .record(&self.stream)?;
                }

                if !lulled_gate {
                    lulled_gate = true;
                    let t = Instant::now();
                    lull(self, Lull::Gate)?;
                    self.phases.add("host work under kernels", t.elapsed());
                }

                let t = Instant::now();
                let mut sums = copy_prefix(&self.scr_surv_sums, &self.stream, blocks as usize)?;
                self.absorb_sync()?;
                let mut total = 0u32;
                for sum in sums.iter_mut() {
                    let count = *sum;
                    *sum = total;
                    total += count;
                }
                #[cfg(feature = "counters")]
                self.hsp_stats.observe_score_gate(iter_num_hits, total);
                self.phases
                    .add("survivor block-sum round trip", t.elapsed());
                // Round 87 (blocks): read the span at this existing sync —
                // no new stream synchronize, no read of an incomplete event
                // (stream order covers the span). Every chunk's span joins
                // the block; settled chunks past T_SETTLE join the settled
                // totals that decide the autotune.
                #[cfg(feature = "ref-loc-buckets")]
                if time_span {
                    let span_ms = self
                        .block_span_start
                        .as_ref()
                        .expect("block span events ensured above")
                        .elapsed_ms(
                            self.block_span_end
                                .as_ref()
                                .expect("block span events ensured above"),
                        )? as f64;
                    let max_hits = self.max_hits;
                    let (closed, decided, inconclusive) = self.bucket_mode.commit_block_span(
                        span_ms,
                        iter_num_hits,
                        max_hits,
                        bucket_block_ms(),
                        bucket_settle_ms(),
                    );
                    if self.timing {
                        if let Some((k, sample)) = &closed {
                            eprintln!(
                                "  ref buckets: block eng={} k={} mode={} chunks={} settled={} hits={} span_ms={:.1} settled_ms={:.1} ns_per_hit={:.3}",
                                self.bucket_eng_id,
                                k,
                                if sample.on { "on" } else { "off" },
                                sample.chunks,
                                sample.settled_chunks,
                                sample.hits,
                                sample.span_ms,
                                sample.ns_per_hit * sample.hits as f64 / 1e6,
                                sample.ns_per_hit,
                            );
                        }
                        if let Some(inc) = &inconclusive {
                            if inc.on {
                                eprintln!(
                                    "  ref buckets: auto -> off (inconclusive: block k={} mode=on span {:.1} ms settled {} chunks)",
                                    inc.k, inc.span_ms, inc.settled_chunks,
                                );
                            } else {
                                eprintln!(
                                    "  ref buckets: dropped block eng={} k={} mode=off (inconclusive; span_ms={:.1} settled={})",
                                    self.bucket_eng_id, inc.k, inc.span_ms, inc.settled_chunks,
                                );
                            }
                        }
                        if let Some(d) = &decided {
                            let r3 = |v: &[f64]| -> Vec<f64> {
                                v.iter().map(|x| (x * 1000.0).round() / 1000.0).collect()
                            };
                            eprintln!(
                                "  ref buckets: auto -> {} (median block ratio {:.3} over 3 pairs; ratios {:?}; off {:?} on {:?} ns/hit)",
                                if d.use_on { "on" } else { "off" },
                                d.median,
                                r3(&d.ratios),
                                r3(&d.off_ns),
                                r3(&d.on_ns),
                            );
                        }
                    }
                }
                if total == 0 {
                    continue;
                }

                let t = Instant::now();
                // SAFETY: emit writes every dense ID, and the materializer
                // writes every dense HSP/status slot before either is read.
                unsafe {
                    reserve(
                        &mut self.buf_survivors,
                        &self.stream,
                        total as usize,
                        &mut self.pipeline_syncs,
                    )?;
                    reserve(
                        &mut self.buf_hsp,
                        &self.stream,
                        total as usize,
                        &mut self.pipeline_syncs,
                    )?;
                    reserve(
                        &mut self.buf_done,
                        &self.stream,
                        total as usize,
                        &mut self.pipeline_syncs,
                    )?;
                }
                self.phases.add("alloc dense outputs", t.elapsed());

                // SAFETY: queued before the emit kernels, which read it; `sums`
                // leaves this block with the total and lives through the done
                // scan's wait on the stream.
                unsafe {
                    upload(
                        &mut self.scr_surv_offsets,
                        &self.stream,
                        &sums,
                        &mut self.pipeline_syncs,
                    )?;
                }
                let stage = self.stage();
                if !bits_pass {
                    unsafe {
                        self.module.emit_survivors(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (blocks, 1, 1),
                                block_dim: (SCAN_BLOCK, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &mut self.buf_flags,
                            &self.scr_surv_offsets,
                            &mut self.buf_survivors,
                            iter_num_hits,
                        )?;
                    }
                }
                // The plain sieve kernels index their bitmask by raw hit id: the
                // set bits are the survivor ids, already ascending.
                // So are the sorted slots of a bucketed chunk whose survivors
                // keep that order (round 116): each is its own id into the
                // sorted anchors.
                #[cfg(feature = "gate-sieve")]
                if bits_pass && (!bucket_pass || slot_order) {
                    // SAFETY: the block prefixes partition the `total` survivor
                    // slots, one per set bit.
                    unsafe {
                        self.module.emit_bits_plain(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (blocks, 1, 1),
                                block_dim: (SCAN_BLOCK, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &self.buf_flags_bits,
                            &self.scr_surv_offsets,
                            &mut self.buf_survivors,
                            flag_words,
                        )?;
                    }
                }
                #[cfg(feature = "ref-loc-buckets")]
                if bucket_pass && !slot_order {
                    // SAFETY: the block prefixes partition the `total` survivor
                    // slots; a set bit implies a sorted slot below `iter_num_hits`,
                    // so every `sorted_idx` read is in bounds.
                    unsafe {
                        self.module.emit_bits(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (blocks, 1, 1),
                                block_dim: (SCAN_BLOCK, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            &self.buf_flags_bits,
                            &self.scr_surv_offsets,
                            &self.buf_sorted_idx,
                            &mut self.buf_survivors,
                            flag_words,
                        )?;
                    }
                }
                self.end_stage(stage, "emit_survivors")?;

                // `emit_bits` emits in ascending SORTED-slot order; `dump_raw`
                // and the census walk show ascending ORIGINAL ids, so the paths
                // that keep the ids restore them. (The output needs neither:
                // `survivor_slots_request`.) One block on the device up to
                // `SORT_MAX`; longer lists go to the host, which is rare enough
                // to have its own ledger row rather than a second kernel.
                #[cfg(feature = "ref-loc-buckets")]
                if bucket_pass && !slot_order {
                    if total <= kernels::SORT_MAX {
                        let stage = self.stage();
                        // SAFETY: `total` ids in `buf_survivors`, sorted in place
                        // by one block that owns the whole range.
                        unsafe {
                            self.module.sort_survivors(
                                &self.stream,
                                LaunchConfig {
                                    grid_dim: (1, 1, 1),
                                    block_dim: (kernels::ONE_BLOCK_THREADS, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                &mut self.buf_survivors,
                                total,
                            )?;
                        }
                        self.end_stage(stage, "sort_survivors")?;
                    } else {
                        let t = Instant::now();
                        let mut ids =
                            copy_prefix(&self.buf_survivors, &self.stream, total as usize)?;
                        self.absorb_sync()?;
                        assert_eq!(
                            ids.len(),
                            total as usize,
                            "survivor count from count_bits does not match the emitted list"
                        );
                        ids.sort_unstable();
                        // SAFETY: `total` u32 into the prefix of a buffer
                        // reserved for at least `total`, and the copy is
                        // synchronized before `ids` is dropped.
                        unsafe {
                            cuda_core::memory::memcpy_htod_async(
                                self.buf_survivors.cu_deviceptr(),
                                ids.as_ptr(),
                                std::mem::size_of_val(ids.as_slice()),
                                self.stream.cu_stream(),
                            )?;
                        }
                        self.stream.synchronize()?;
                        self.pipeline_syncs += 1;
                        // Round 87 (blocks): the host fallback runs untimed —
                        // the span ends at the count.
                        let host_sort = t.elapsed();
                        self.phases.add("survivor host sort", host_sort);
                    }
                    // The restored order is what byte-identity depends on but
                    // cannot see: a valid permutation of the right ids still
                    // reorders `dump_raw`, the census walk and dedup ties.
                    if order_check_limit() > 0 {
                        static SORT_CHECK_DONE: std::sync::atomic::AtomicU32 =
                            std::sync::atomic::AtomicU32::new(0);
                        let check_c =
                            SORT_CHECK_DONE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if check_c < order_check_limit() {
                            self.stream.synchronize()?;
                            let ids =
                                copy_prefix(&self.buf_survivors, &self.stream, total as usize)?;
                            let mut detail = String::from("none");
                            let mut ok = true;
                            for (p, w) in ids.windows(2).enumerate() {
                                if w[0] >= w[1] {
                                    ok = false;
                                    detail = format!("p={p} id={} next={}", w[0], w[1]);
                                    break;
                                }
                            }
                            if let Some(&last) = ids.last()
                                && last >= iter_num_hits
                            {
                                ok = false;
                                detail = format!("last={last} n={iter_num_hits}");
                            }
                            if !ok {
                                eprintln!(
                                    "#bucket-check-violation chunk={check_c} survivor_order {detail}"
                                );
                            }
                            eprintln!(
                                "#survivor-order-check chunk={check_c} n={total} path={} ascending={}",
                                if total <= kernels::SORT_MAX {
                                    "device"
                                } else {
                                    "host"
                                },
                                if ok { "ok" } else { "FAIL" },
                            );
                        }
                    }
                }
                (total, sums)
            };
            #[cfg(not(feature = "dense-anchors"))]
            let materializer_hits = iter_num_hits;

            let t = Instant::now();
            // Two words per materialized hit only under `counters`; otherwise
            // the kernel receives one unused element.
            let stats_len = if cfg!(feature = "counters") {
                2 * materializer_hits as usize
            } else {
                1
            };
            #[cfg(feature = "counters")]
            let mut d_stats = DeviceBuffer::<u64>::zeroed(&self.stream, stats_len)?;
            #[cfg(not(feature = "counters"))]
            let _ = stats_len;
            if !self.async_stages {
                self.stream.synchronize()?;
                self.stage_syncs += 1;
            }
            self.phases.add("alloc materializer buffers", t.elapsed());
            // The peak can only have risen if something was allocated since the
            // last sample, so the driver is asked then and not once per chunk.
            let epoch = ALLOC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
            if epoch != self.mem_epoch_seen {
                self.mem_epoch_seen = epoch;
                let (free, total) = device_memory();
                self.peak_used = self.peak_used.max(total.saturating_sub(free));
            }

            #[cfg(feature = "counters")]
            self.hsp_stats.set_strand(rev);
            let query = if rev {
                &self.query_rc_seq
            } else {
                &self.query_seq
            };
            #[cfg(all(feature = "dense-anchors", feature = "ref-loc-buckets"))]
            let anchor_ptr = if slot_order {
                // The survivor ids are sorted slots.
                self.buf_sorted_anchor.cu_deviceptr() as *const u64
            } else {
                self.buf_anchor.cu_deviceptr() as *const u64
            };
            #[cfg(all(feature = "dense-anchors", not(feature = "ref-loc-buckets")))]
            let anchor_ptr = self.buf_anchor.cu_deviceptr() as *const u64;
            #[cfg(not(feature = "dense-anchors"))]
            let anchor_ptr = core::ptr::null::<u64>();
            #[cfg(feature = "dense-anchors")]
            let survivor_ptr = self.buf_survivors.cu_deviceptr() as *const u32;
            #[cfg(not(feature = "dense-anchors"))]
            let survivor_ptr = core::ptr::null::<u32>();
            let stage = self.stage();
            #[cfg(feature = "counters")]
            let stats = &mut d_stats;
            #[cfg(not(feature = "counters"))]
            let stats = &mut self.scr_stats;
            // SAFETY: one warp per work item; `hsp`/`done` are sized to the
            // materializer count and the dense pointers cover every emitted ID.
            unsafe {
                self.module.find_hsps(
                    &self.stream,
                    LaunchConfig {
                        // Round 126: one warp takes one survivor, so a batch's
                        // few hundred need a few hundred warps, not the 65,536
                        // of the full grid (a block past the survivors copies
                        // the matrix, waits at the barrier and leaves). The
                        // result of a survivor does not depend on the grid.
                        grid_dim: (
                            if tail_lean() {
                                self.hsp_blocks
                                    .min(materializer_hits.div_ceil(NUM_WARPS as u32).max(1))
                            } else {
                                self.hsp_blocks
                            },
                            1,
                            1,
                        ),
                        block_dim: (HSP_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &self.ref_seq,
                    query,
                    self.ref_len,
                    self.query_len,
                    &self.sub_mat,
                    self.noentropy,
                    self.xdrop,
                    self.hspthresh,
                    materializer_hits,
                    LOG4,
                    anchor_ptr,
                    survivor_ptr,
                    &mut self.buf_hsp,
                    &mut self.buf_done,
                    stats,
                )?;
            }
            self.end_stage(stage, "find_hsps")?;

            // S0 survivor audit. Must run here: `buf_done` holds one 0/1
            // accepted flag per survivor slot only until the done-scan below
            // overwrites it in place with the cumulative. Env-gated and off the
            // timed path; folds to scalars and drops every buffer.
            #[cfg(feature = "dense-anchors")]
            if self.census.is_some() {
                let n = materializer_hits as usize;
                let survivors = copy_prefix(&self.buf_survivors, &self.stream, n)?;
                let accepted = copy_prefix(&self.buf_done, &self.stream, n)?;
                let hsps = copy_prefix(&self.buf_hsp, &self.stream, n)?;
                self.absorb_sync()?;
                let lo = start_seed_index as usize;
                let hi = lo + iter_num_seeds as usize;
                if let Some(c) = self.census.as_mut() {
                    let counts = hit_counts.as_deref().expect("census requested hit counts");
                    c.ingest_survivors(
                        &counts[lo..hi],
                        &survivors,
                        &accepted,
                        &hsps,
                        std::mem::size_of_val(counts) as u64,
                        &mut out.audit,
                    );
                }
            }

            #[cfg(feature = "counters")]
            {
                let raw = d_stats.to_host_vec(&self.stream)?;
                self.absorb_sync()?;
                self.hsp_stats.observe(&raw);
            }
            #[cfg(feature = "counters")]
            let _ = &d_stats;

            // Round 126: a chunk with few survivors does not scan, round-trip,
            // offset and compact their accepted flags to get its HSPs: the
            // host copies the survivors' records as `find_hsps` left them and
            // keeps the accepted ones. An accepted record has a positive score
            // and a rejected one a zero (`find_hsps` stores `total * entropy`
            // truncated when that reaches the threshold, else 0; with a
            // threshold of 2 or more the stored score is at least 1), and the
            // records are in slot order, which is `compress_output`'s.
            if materializer_hits <= SMALL_TAIL && self.hspthresh >= 2 && tail_lean() {
                let t = Instant::now();
                let mut anchors =
                    copy_prefix(&self.buf_hsp, &self.stream, materializer_hits as usize)?;
                self.absorb_sync()?;
                anchors.retain(|h| h.score > 0);
                self.phases.add("D->H survivors' records", t.elapsed());
                tail_totals::SMALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if anchors.is_empty() {
                    continue;
                }
                out.raw_hsps += anchors.len() as u32;
                let t = Instant::now();
                if self.dump_raw {
                    out.raw.extend_from_slice(&anchors);
                }
                if defer_dedup && cfg!(not(feature = "counters")) {
                    out.chunks.push(anchors);
                    continue;
                }
                #[cfg(feature = "counters")]
                crate::hsp::dedup_and_order(&mut anchors, Some(&mut self.groups));
                #[cfg(not(feature = "counters"))]
                crate::hsp::dedup_and_order(&mut anchors, None);
                out.hsps.extend_from_slice(&anchors);
                self.phases.add("host sort + dedup", t.elapsed());
                continue;
            }
            tail_totals::SCANNED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Scan the done flags in place on the device with the
            // same two-kernel machinery already accepted for the hit counts, so
            // only the per-block sums (`materializer_hits/256` u32) cross the bus
            // instead of the whole flag array in each direction. `num_anchors`
            // falls out of the block-sum scan, and `compress_output` is
            // unchanged — it still reads a full inclusive scan.
            let done_blocks = materializer_hits.div_ceil(SCAN_BLOCK);
            // SAFETY: `scan_blocks` writes one sum for every launched block.
            unsafe {
                reserve(
                    &mut self.scr_done_sums,
                    &self.stream,
                    done_blocks as usize,
                    &mut self.pipeline_syncs,
                )?;
            }
            let stage = self.stage();
            // SAFETY: one element per thread over `materializer_hits`, one sum slot
            // per block.
            unsafe {
                self.module.scan_blocks(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: (done_blocks, 1, 1),
                        block_dim: (SCAN_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut self.buf_done,
                    &mut self.scr_done_sums,
                    materializer_hits,
                )?;
            }
            self.end_stage(stage, "scan_blocks (done)")?;

            let t = Instant::now();
            // REQUIRED: `num_anchors` sizes the reduced buffer and decides whether
            // this chunk emits anything at all.
            let mut sums = copy_prefix(&self.scr_done_sums, &self.stream, done_blocks as usize)?;
            self.absorb_sync()?;
            let mut acc = 0u32;
            for s in sums.iter_mut() {
                let total = *s;
                *s = acc;
                acc += total;
            }
            let num_anchors = acc;
            self.phases.add("done block-sum round trip", t.elapsed());
            if num_anchors == 0 {
                continue;
            }
            out.raw_hsps += num_anchors;

            // SAFETY: queued before `add_block_offsets`, which reads it; `sums`
            // lives to the end of this iteration, past the wait on the output.
            unsafe {
                upload(
                    &mut self.scr_done_offsets,
                    &self.stream,
                    &sums,
                    &mut self.pipeline_syncs,
                )?;
            }
            let stage = self.stage();
            // SAFETY: same geometry as `scan_blocks`, so block `b` reads offset `b`.
            unsafe {
                self.module.add_block_offsets(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: (done_blocks, 1, 1),
                        block_dim: (SCAN_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut self.buf_done,
                    &self.scr_done_offsets,
                    materializer_hits,
                )?;
            }
            self.end_stage(stage, "add_block_offsets (done)")?;

            let t = Instant::now();
            // SAFETY: `compress_output` writes every one of the `num_anchors`
            // slots (one per done flag) before the copy below reads them.
            unsafe {
                reserve(
                    &mut self.buf_reduced,
                    &self.stream,
                    num_anchors as usize,
                    &mut self.pipeline_syncs,
                )?;
            }
            if !self.async_stages {
                self.stream.synchronize()?;
                self.stage_syncs += 1;
            }
            self.phases.add("alloc reduced", t.elapsed());

            let stage = self.stage();
            // SAFETY: grid-stride over `materializer_hits`; the scan guarantees
            // every stored index is below `num_anchors`.
            unsafe {
                self.module.compress_output(
                    &self.stream,
                    elementwise(),
                    &self.buf_done,
                    &self.buf_hsp,
                    &mut self.buf_reduced,
                    materializer_hits,
                )?;
            }
            self.end_stage(stage, "compress_output")?;

            let t = Instant::now();
            // REQUIRED: this is the output. It also drains the stream, which is
            // what makes the per-chunk temporaries safe to drop below.
            let mut anchors = copy_prefix(&self.buf_reduced, &self.stream, num_anchors as usize)?;
            self.absorb_sync()?;
            self.phases.add("D->H anchors", t.elapsed());

            let t = Instant::now();
            if self.dump_raw {
                out.raw.extend_from_slice(&anchors);
            }
            // With `counters` the dedup feeds the engine's group statistics, so
            // it stays here.
            if defer_dedup && cfg!(not(feature = "counters")) {
                out.chunks.push(anchors);
                continue;
            }
            #[cfg(feature = "counters")]
            crate::hsp::dedup_and_order(&mut anchors, Some(&mut self.groups));
            #[cfg(not(feature = "counters"))]
            crate::hsp::dedup_and_order(&mut anchors, None);
            out.hsps.extend_from_slice(&anchors);
            self.phases.add("host sort + dedup", t.elapsed());
        }

        // The copy stream may not overwrite this slot until every kernel that read
        // it has finished. Recorded even though today's boundary drain already
        // guarantees it, so the invariant survives relaxing that drain.
        self.compute_done[slot].record(&self.stream)?;

        // Boundary invariant: `seed_and_filter` returns with the stream drained.
        // Unresolved stages are the exact signal that work may still be queued —
        // every genuine sync resolves them — and the caller reuses the host seed
        // slot and may free device buffers as soon as we return.
        if !self.pending.is_empty() {
            self.sync_pipeline()?;
        }
        // The upload ran concurrently, so its duration belongs in the overlapped
        // section of the table, not in `accounted`.
        if self.async_seed_copy
            && self.timing
            && num_seeds > 0
            && let Some(start) = &self.copy_start[slot]
        {
            let ms = start.elapsed_ms(&self.copy_ready[slot])? as f64;
            self.phases.add_overlapped_ms("H->D seeds (standalone)", ms);
        }
        Ok(out)
    }

    /// Round 110: expands a chunk's seeds into `buf_anchor` with one warp per
    /// run of 32 seeds (thread `t` owns seed slot `t`), from the start indices
    /// `find_num_hits_lo` left in `buf_seed_lo`.
    #[cfg(feature = "find-hits-groups")]
    fn launch_find_hits_groups(
        &mut self,
        slot: usize,
        start_seed_index: u32,
        start_hit_index: u32,
        num_seeds: u32,
    ) -> Result<(), DriverError> {
        // SAFETY: every store lands inside `0..iter_num_hits` by construction of
        // the prefix offsets, as for the other `find_hits` kernels.
        unsafe {
            self.module.find_hits_groups(
                &self.stream,
                LaunchConfig {
                    grid_dim: (num_seeds.div_ceil(BLOCK_SIZE), 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.pos_table,
                &self.buf_seed_lo,
                &self.seed_slots[slot],
                self.seed_size,
                &self.buf_hit_num,
                &mut self.buf_anchor,
                start_seed_index,
                start_hit_index,
                num_seeds,
            )
        }
    }

    /// Round 110, `HSPZ_FIND_HITS_GROUPS=check`: a per-seed kernel has just
    /// written this chunk's anchors; expand the seeds again by groups into a
    /// zeroed scratch buffer and compare every anchor on the host.
    #[cfg(feature = "find-hits-groups")]
    fn check_find_hits_groups(
        &mut self,
        slot: usize,
        start_seed_index: u32,
        start_hit_index: u32,
        num_seeds: u32,
        num_hits: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::atomic::Ordering::Relaxed;
        let n = num_hits as usize;
        self.sync_pipeline()?;
        let want = copy_prefix(&self.buf_anchor, &self.stream, n)?;
        // Zeroed, so a slot the group kernel never wrote cannot pass on what was
        // there before (no anchor is zero: `ref_loc >= seed_size`).
        let mut scratch = DeviceBuffer::<u64>::zeroed(&self.stream, n)?;
        std::mem::swap(&mut self.buf_anchor, &mut scratch);
        let stage = self.stage();
        let launched =
            self.launch_find_hits_groups(slot, start_seed_index, start_hit_index, num_seeds);
        let ended = self.end_stage(stage, "find_hits (groups)");
        let synced = self.sync_pipeline();
        std::mem::swap(&mut self.buf_anchor, &mut scratch);
        launched?;
        ended?;
        synced?;
        let got = copy_prefix(&scratch, &self.stream, n)?;
        if let Some(p) = (0..n).find(|&p| got[p] != want[p]) {
            return Err(format!(
                "find-hits-groups check: anchor {p} of {n} (chunk seeds {start_seed_index}+{num_seeds}): \
                 {:#018x} by groups, {:#018x} by the per-seed kernel",
                got[p], want[p]
            )
            .into());
        }
        find_hits_groups_totals::CHECKED_CHUNKS.fetch_add(1, Relaxed);
        find_hits_groups_totals::CHECKED_HITS.fetch_add(n as u64, Relaxed);
        Ok(())
    }

    fn stage(&self) -> Stage {
        Stage::begin(&self.stream, self.timing)
    }

    /// Ends a GPU stage: records its end event, counts the launch, and — only
    /// under `--sync-stages` — waits for it.
    ///
    /// The default path enqueues and returns. Stream order already guarantees
    /// that the next kernel sees this one's writes, so the host has nothing to
    /// wait for.
    fn end_stage(&mut self, stage: Stage, name: &'static str) -> Result<(), DriverError> {
        let done = stage.finish(name)?;
        self.pending.push(done);
        self.launches += 1;
        if !self.async_stages {
            self.stream.synchronize()?;
            self.stage_syncs += 1;
            self.resolve_pending()?;
        }
        Ok(())
    }

    /// Reads every recorded stage's event pair into `phases`.
    ///
    /// Only correct after a synchronization that covers those events, which is
    /// why every caller is a sync point.
    fn resolve_pending(&mut self) -> Result<(), DriverError> {
        for p in std::mem::take(&mut self.pending) {
            let gpu_ms = match &p.events {
                Some((start, end)) => start.elapsed_ms(end)?,
                None => 0.0,
            };
            // Round 71: pair the previous stage's end with this stage's start. Both are
            // on one stream, so the delta is GPU-timeline idle — the bubble the host put
            // there. A negative or absurd reading would mean the events are unordered,
            // so anything outside [0, 1000] ms is dropped rather than trusted.
            // Intervals above 1,000 ms are unit-transition work (query pack, ref
            // prep) counted separately so they stay visible instead of silent.
            if let (Some(prev), Some((start, _))) = (self.last_end.as_ref(), p.events.as_ref())
                && let Ok(gap) = prev.elapsed_ms(start)
            {
                if (0.0f32..=1000.0f32).contains(&gap) {
                    self.gap_ms += gap;
                    self.gap_n += 1;
                    if gap > self.gap_max {
                        self.gap_max = gap;
                        self.gap_max_pair = (self.last_name, p.name);
                    }
                    let key = (self.last_name, p.name);
                    match self.gap_pairs.iter_mut().find(|e| e.0 == key) {
                        Some(e) => {
                            e.1 += gap;
                            e.2 += 1;
                        }
                        None => self.gap_pairs.push((key, gap, 1)),
                    }
                } else if gap > 1000.0 {
                    self.gap_long_n += 1;
                    self.gap_long_ms += gap;
                }
            }
            if let Some((_, end)) = p.events {
                self.last_end = Some(end);
                self.last_name = p.name;
            }
            self.phases.add_gpu(p.name, p.host, gpu_ms);
        }
        Ok(())
    }

    /// Round 71: clear the gap accumulator so a reading covers one timed pass only.
    ///
    /// Without this the accumulator spans the cold pass — whose ~1 s of seed-table build
    /// runs with the GPU idle — and a warm-median wall is then quoted against gaps that
    /// mostly came from setup. That was the first reading's 2.5x over-count.
    ///
    /// `last_end` is cleared too: the first pair of a new pass would otherwise reach back
    /// across the reset and charge this pass for the previous one's tail.
    pub fn reset_gaps(&mut self) {
        self.gap_ms = 0.0;
        self.gap_n = 0;
        self.gap_max = 0.0;
        self.gap_max_pair = ("", "");
        self.gap_pairs.clear();
        self.gap_long_n = 0;
        self.gap_long_ms = 0.0;
        self.last_end = None;
        self.last_name = "";
    }

    /// Round 72: per-pair gap totals, largest first.
    pub fn gap_pairs(&self) -> Vec<((&'static str, &'static str), f32, u64)> {
        let mut v = self.gap_pairs.clone();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// Intervals discarded by the gap accounting because they exceed 1,000 ms
    /// (unit-transition work), as `(count, summed ms)`.
    pub fn discarded_gaps(&self) -> (u64, f32) {
        (self.gap_long_n, self.gap_long_ms)
    }

    /// Round 71: `(summed gap ms, pairs measured, largest gap, its stage pair)`.
    pub fn stage_gaps(&self) -> (f32, u64, f32, (&'static str, &'static str)) {
        (self.gap_ms, self.gap_n, self.gap_max, self.gap_max_pair)
    }

    /// Engine-end autotune ledger: if still in `Auto` (the chunks ran out
    /// before 6 blocks), the count of closed blocks so the log says why
    /// there is no decision. `None` once forced or decided.
    #[cfg(feature = "ref-loc-buckets")]
    pub fn bucket_undecided_blocks(&self) -> Option<usize> {
        self.bucket_mode.undecided_blocks()
    }

    /// Prints the engine-end ledger line when still undecided. Call once per
    /// engine after its last chunk, under `--time` only via `self.timing`.
    pub fn finish_bucket_autotune(&self) {
        #[cfg(feature = "ref-loc-buckets")]
        {
            if let Some(k) = self.bucket_undecided_blocks()
                && self.timing
            {
                eprintln!("ref buckets: auto undecided (engine ended after {k} blocks)");
            }
        }
    }

    /// A genuine host dependency: the host is about to read a device
    /// result, free a buffer a queued kernel touches, or return to the caller.
    fn sync_pipeline(&mut self) -> Result<(), DriverError> {
        host_trace::timed(&host_trace::BLOCKED_NS, || self.stream.synchronize())?;
        self.pipeline_syncs += 1;
        self.resolve_pending()
    }

    /// Same accounting for a wait that a blocking copy already performed —
    /// `to_host_vec` and `from_host` both synchronize the stream inside
    /// cuda-core, so calling `sync_pipeline` after one would wait twice.
    fn absorb_sync(&mut self) -> Result<(), DriverError> {
        self.pipeline_syncs += 1;
        self.resolve_pending()
    }

    /// Host waits that existed only to measure or to serialise already-ordered
    /// stages. The mechanism gate: this must fall to zero.
    pub fn stage_syncs(&self) -> u64 {
        self.stage_syncs
    }

    /// Phase 3 mechanism: uploads issued, and uploads that had not completed when
    /// their compute needed them. `stalls == 0` means every DMA hid behind compute.
    pub fn seed_copy_stats(&self) -> (u64, u64) {
        (self.seed_uploads, self.seed_copy_stalls)
    }

    /// Host waits on a real device dependency. Roughly constant across the A/B —
    /// if this moves instead, the mechanism did something other than intended.
    pub fn pipeline_syncs(&self) -> u64 {
        self.pipeline_syncs
    }
}

/// Lifecycle counters for the reference-scoped contract.
///
/// Not statistics: the executor asserts these, because the failure they guard
/// against — rebuilding or re-uploading the reference index per work unit instead
/// of per reference visit — is invisible in output and only shows up as ~1 GB of
/// extra H->D traffic per avoided reuse. Under the round-90 unit partition one
/// visit is one reference build/upload, so `visits` is the bin count on the
/// whole-bin path and the visit count on the unit path.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Lifecycle {
    pub seed_table_builds: u32,
    /// Reference bins loaded from an on-disk index (`run --index DIR`) instead
    /// of built. A visit is either built or loaded, never both, so the visit
    /// invariant is `builds + loads == visits`.
    pub index_loads: u32,
    pub engine_creations: u32,
    pub reference_uploads: u32,
    pub query_swaps: u32,
    pub work_units_executed: u32,
}

impl Lifecycle {
    /// Checks the lifecycle invariants against a plan's shape.
    ///
    /// Returns the first violation as a message rather than panicking, so a
    /// caller can report it alongside the rest of a profile.
    pub fn check(&self, visits: u32, work_units: u32) -> Result<(), String> {
        let want = [
            (
                "seed_table_builds+index_loads",
                self.seed_table_builds + self.index_loads,
                visits,
            ),
            ("engine_creations", self.engine_creations, visits),
            ("reference_uploads", self.reference_uploads, visits),
            ("query_swaps", self.query_swaps, work_units),
            ("work_units_executed", self.work_units_executed, work_units),
        ];
        for (name, got, expect) in want {
            if got != expect {
                return Err(format!(
                    "lifecycle: {name} = {got}, expected {expect} \
                     (visits {visits}, work_units {work_units})"
                ));
            }
        }
        Ok(())
    }
}

/// `logf(4.0f)` widened to double. KegAlign divides the entropy by `log(4.0f)`,
/// which picks the single-precision overload, so the divisor is the float
/// rounding of ln 4 — not `std::f64::consts::LN_2 * 2`.
const LOG4: f64 = 1.3862943649291992;

/// [`DeviceBuffer::zeroed`] without the memset.
///
/// Every use has a producer that overwrites its complete active range before a
/// consumer reads it: hit expansion writes anchors, the gate writes flags, and
/// the materializer writes HSP/status records. Zeroing those buffers is dead
/// work. cuda-oxide's own
/// `uninitialized_async` allocates with `cuMemAllocAsync`, which ZLUDA does not
/// implement (DriverError 801), so this is `zeroed`'s `malloc_sync` without its
/// `memset_d8_async`.
///
/// # Safety
///
/// The producing kernel must write every element before anything reads it.
unsafe fn uninitialized<T>(
    stream: &CudaStream,
    len: usize,
) -> Result<DeviceBuffer<T>, DriverError> {
    let ctx = stream.context().clone();
    let bytes = len.checked_mul(size_of::<T>()).ok_or(DriverError(
        cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
    ))?;
    // cuMemAlloc rejects zero-byte requests, so an empty buffer is a null
    // pointer that `Drop` ignores — the same representation `zeroed` uses.
    if bytes == 0 {
        return Ok(unsafe { DeviceBuffer::from_raw_parts(0, len, ctx) });
    }
    let ptr = unsafe { cuda_core::memory::malloc_sync(bytes)? };
    ALLOC_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(unsafe { DeviceBuffer::from_raw_parts(ptr, len, ctx) })
}

/// What the sieve's guard tells one gate launch to do.
#[cfg(feature = "gate-sieve")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuardStep {
    /// Run the sieve and count its candidates.
    Sample,
    /// Run the sieve.
    Sieve,
    /// Run the plain gate: the last sample rejected too little.
    Plain,
}

/// Keeps the sieve from costing more than it saves. Where the bound rejects
/// little (long unmasked satellite arrays, self-alignment) a hit pays the bound
/// and then exact scoring, and the bound's own time is 11-25% of exact scoring.
/// Every `period`th gate launch is a sample: it runs the sieve and counts the
/// candidates. If more than [`SieveGuard::HOLD_ABOVE`] of its hits were
/// candidates, the launches up to the next sample run the plain gate. Both
/// gates are exact, so the output never depends on the guard.
#[cfg(feature = "gate-sieve")]
struct SieveGuard {
    /// Launches per sample; 0 switches the guard off (`HSPZ_GATE_SIEVE_GUARD`).
    period: u32,
    tick: u32,
    hold: bool,
}

#[cfg(feature = "gate-sieve")]
impl SieveGuard {
    /// Candidate fraction above which the plain gate is the cheaper one. The
    /// sieve breaks even where exact scoring of the candidates plus the bound
    /// equals exact scoring of everything, a fraction of 0.75-0.89 on the cards
    /// measured; half leaves a margin.
    const HOLD_ABOVE: f64 = 0.5;
    /// A sample this small says little about the launches that follow.
    const MIN_HITS: u64 = 1 << 16;

    fn new(period: u32) -> Self {
        Self {
            period,
            tick: 0,
            hold: false,
        }
    }

    fn next_batch(&mut self) -> GuardStep {
        if self.period == 0 {
            return GuardStep::Sieve;
        }
        let step = if self.tick == 0 {
            // A hold lasts until the next sample and no longer: a sample too
            // small to decide (`observe` leaves it alone) must not keep the
            // plain gate on for another period.
            self.hold = false;
            GuardStep::Sample
        } else if self.hold {
            GuardStep::Plain
        } else {
            GuardStep::Sieve
        };
        self.tick = (self.tick + 1) % self.period;
        step
    }

    /// Whether [`next_batch`](Self::next_batch) would say `Plain`, without
    /// taking the step.
    #[cfg(all(feature = "find-hits-groups", feature = "ref-loc-buckets"))]
    fn next_is_plain(&self) -> bool {
        self.period != 0 && self.tick != 0 && self.hold
    }

    fn observe(&mut self, candidates: u64, hits: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        if hits < Self::MIN_HITS {
            return;
        }
        self.hold = candidates as f64 > Self::HOLD_ABOVE * hits as f64;
        sieve_totals::GUARD_SAMPLES.fetch_add(1, Relaxed);
        sieve_totals::GUARD_HOLDS.fetch_add(u64::from(self.hold), Relaxed);
    }
}

/// `HSPZ_GATE_SIEVE_GUARD`: gate launches per guard sample (default 32), `0` for
/// no guard. Read once; anything else is reported and the default used.
#[cfg(feature = "gate-sieve")]
fn sieve_guard_period() -> u32 {
    const DEFAULT: u32 = 32;
    static PERIOD: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *PERIOD.get_or_init(|| match std::env::var("HSPZ_GATE_SIEVE_GUARD") {
        Err(_) => DEFAULT,
        Ok(v) => v.trim().parse().unwrap_or_else(|_| {
            eprintln!("warning: HSPZ_GATE_SIEVE_GUARD={v:?} is not an integer; using {DEFAULT}");
            DEFAULT
        }),
    })
}

/// `gate-sieve`: what the engine holds for the lane-local bound in front of the
/// score gate (`gpu/sieve.rs`): the packed planes of the reference and of both
/// query strands, and this unit's class scores.
#[cfg(feature = "gate-sieve")]
struct GateSieve {
    /// `HSPZ_GATE_SIEVE=0` keeps every launch on the plain gate.
    requested: bool,
    /// Which kernels load the reference groups through shared memory instead of
    /// every lane loading its own: all (`HSPZ_GATE_SIEVE=gather`), none
    /// (`scatter`, the default), or `None` for `auto` — gathered on the plain
    /// gate, where the loads miss the L2, and per lane under the bucket pass.
    gather: Option<bool>,
    /// `HSPZ_GATE_SIEVE_COUNT=1`: read every batch's candidate words back and
    /// report the passing fraction. A diagnostic: it waits on every batch.
    count: bool,
    sub_mat: Vec<i32>,
    ref_has_x: bool,
    ref_pk: DeviceBuffer<u64>,
    qry_pk: [DeviceBuffer<u64>; 2],
    qry_bad: [DeviceBuffer<u32>; 2],
    /// `Classes::kernel_args` for the unit in place; `None` runs the plain gate.
    args: Option<(i32, i32, i32, i32)>,
    classes: Option<sieve::Classes>,
    /// Round 111: how far an undecided side of the bound is carried
    /// (`HSPZ_SIEVE_WORDS=R,L`).
    words: sieve::Words,
    /// Round 117: whether the bound reads the query's mask plane
    /// (`HSPZ_GATE_SIEVE_MASK`; without it the plane is not built).
    masked: bool,
    /// The state last printed under `--time`, so that a run prints it once per
    /// change rather than once per unit.
    said: String,
    cand_bits: DeviceBuffer<u32>,
    guard: SieveGuard,
    /// `HSPZ_GATE_SIEVE_CHECK=1`: host copies of both sequences and their
    /// planes, to recompute every hit on the host — the lane's decision, and the
    /// exact gate's — and compare both with the device.
    host: Option<SieveHost>,
}

#[cfg(feature = "gate-sieve")]
#[derive(Default)]
struct SieveHost {
    ref_enc: Vec<u8>,
    ref_pk: Vec<u64>,
    qry_enc: [Vec<u8>; 2],
    qry_pk: [Vec<u64>; 2],
    qry_bad: [Vec<u32>; 2],
}

/// Process-wide tallies of the sieve diagnostics, printed with the run's counts.
#[cfg(feature = "gate-sieve")]
pub mod sieve_totals {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub(super) static HITS: AtomicU64 = AtomicU64::new(0);
    pub(super) static CANDIDATES: AtomicU64 = AtomicU64::new(0);
    pub(super) static CHECKED: AtomicU64 = AtomicU64::new(0);
    pub(super) static KEPT: AtomicU64 = AtomicU64::new(0);
    pub(super) static CANDIDATE_MISMATCHES: AtomicU64 = AtomicU64::new(0);
    pub(super) static FLAG_MISMATCHES: AtomicU64 = AtomicU64::new(0);
    pub(super) static KEPT_BUT_REJECTED: AtomicU64 = AtomicU64::new(0);
    pub(super) static PLANE_MISMATCHES: AtomicU64 = AtomicU64::new(0);
    pub(super) static GUARD_SAMPLES: AtomicU64 = AtomicU64::new(0);
    pub(super) static GUARD_HOLDS: AtomicU64 = AtomicU64::new(0);

    /// The lines for the run report; empty unless a diagnostic ran or the guard
    /// held the plain gate.
    pub fn report() -> Vec<String> {
        let mut out = Vec::new();
        let (hits, cands) = (HITS.load(Relaxed), CANDIDATES.load(Relaxed));
        if hits > 0 {
            out.push(format!(
                "#sieve candidates: {cands} of {hits} hits ({:.4}%)",
                cands as f64 / hits as f64 * 100.0
            ));
        }
        let (samples, holds) = (GUARD_SAMPLES.load(Relaxed), GUARD_HOLDS.load(Relaxed));
        if holds > 0 {
            out.push(format!(
                "#sieve guard: the bound passed more than half of the hits in {holds} of {samples} \
                 sampled batches; the plain gate ran after each of them"
            ));
        }
        let checked = CHECKED.load(Relaxed);
        if checked > 0 {
            out.push(format!(
                "#sieve check: {checked} hits, {} kept by the exact model; candidate mismatches {}, \
                 flag mismatches {}, kept but rejected {}, plane word mismatches {}",
                KEPT.load(Relaxed),
                CANDIDATE_MISMATCHES.load(Relaxed),
                FLAG_MISMATCHES.load(Relaxed),
                KEPT_BUT_REJECTED.load(Relaxed),
                PLANE_MISMATCHES.load(Relaxed),
            ));
        }
        out
    }
}

/// `HSPZ_GATE_SIEVE` as (sieve requested, gathered reference): unset, `1` or
/// `scatter` run the sieve with every lane loading its own reference groups,
/// `gather` gathers them through shared memory, `auto` gathers on the plain
/// gate only, `0` keeps the plain gate. Read once; any other value is reported
/// and runs the default arm.
#[cfg(feature = "gate-sieve")]
fn sieve_request() -> (bool, Option<bool>) {
    static REQUEST: std::sync::OnceLock<(bool, Option<bool>)> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| match std::env::var("HSPZ_GATE_SIEVE").as_deref() {
        Err(_) | Ok("1") | Ok("scatter") => (true, Some(false)),
        Ok("gather") => (true, Some(true)),
        Ok("auto") => (true, None),
        Ok("0") => (false, Some(false)),
        Ok(other) => {
            eprintln!(
                "warning: HSPZ_GATE_SIEVE={other:?} is not 0, 1, scatter, gather or auto; using scatter"
            );
            (true, Some(false))
        }
    })
}

/// Round 117: whether the bound reads the query's mask plane.
///
/// Unset or `HSPZ_GATE_SIEVE_MASK=0`: the query is packed with
/// `sieve::QRY_FILL`, no mask plane is built for it, and only a drop certifies a
/// side of the bound. `1`: the plane of rounds 104 to 116, where a query base
/// outside `ACGT` ends the bound's walk. The gathered arms keep the plane, so
/// asking for them keeps it everywhere.
#[cfg(feature = "gate-sieve")]
fn sieve_masked() -> bool {
    static MASKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *MASKED.get_or_init(|| {
        let asked = match std::env::var("HSPZ_GATE_SIEVE_MASK").as_deref() {
            Err(_) => None,
            Ok("0") => Some(false),
            Ok("1") => Some(true),
            Ok(other) => {
                eprintln!("warning: HSPZ_GATE_SIEVE_MASK={other:?} is not 0 or 1; using 0");
                None
            }
        };
        if sieve_request().1 != Some(false) {
            if asked == Some(false) {
                eprintln!(
                    "warning: HSPZ_GATE_SIEVE_MASK=0 has no gathered arm; the mask plane stays \
                     (HSPZ_GATE_SIEVE=gather or auto)"
                );
            }
            return true;
        }
        asked.unwrap_or(false)
    })
}

/// `HSPZ_SIEVE_WORDS=R,L`: the words (32 bases each) the bound may read on the
/// right and on the left of a hit. Every lane reads 2 and 3; a side that has not
/// certified by then is carried on up to `R` and `L`. Unset is the longest
/// window, 4 and 5 (round 111); `2,3` is the static window of rounds 104 to 110.
/// A value outside those ranges is reported once and ignored.
#[cfg(feature = "gate-sieve")]
fn sieve_words() -> sieve::Words {
    const DEFAULT: sieve::Words = (sieve::RIGHT_WORDS_MAX, sieve::LEFT_WORDS_MAX);
    static WORDS: std::sync::OnceLock<sieve::Words> = std::sync::OnceLock::new();
    *WORDS.get_or_init(|| {
        let Ok(text) = std::env::var("HSPZ_SIEVE_WORDS") else {
            return DEFAULT;
        };
        let parsed = text.split_once(',').and_then(|(r, l)| {
            let (r, l) = (r.trim().parse::<u32>().ok()?, l.trim().parse::<u32>().ok()?);
            ((sieve::RIGHT_WORDS..=sieve::RIGHT_WORDS_MAX).contains(&r)
                && (sieve::LEFT_WORDS..=sieve::LEFT_WORDS_MAX).contains(&l))
            .then_some((r, l))
        });
        parsed.unwrap_or_else(|| {
            eprintln!(
                "warning: HSPZ_SIEVE_WORDS={text:?} is not R,L with R in {}..={} and L in {}..={}; \
                 using {},{}",
                sieve::RIGHT_WORDS,
                sieve::RIGHT_WORDS_MAX,
                sieve::LEFT_WORDS,
                sieve::LEFT_WORDS_MAX,
                DEFAULT.0,
                DEFAULT.1,
            );
            DEFAULT
        })
    })
}

#[cfg(feature = "gate-sieve")]
impl GateSieve {
    fn new(stream: &CudaStream, sub_mat: &[i32]) -> Result<Self, DriverError> {
        let flag = |name: &str| matches!(std::env::var(name).as_deref(), Ok("1"));
        let check = flag("HSPZ_GATE_SIEVE_CHECK");
        // SAFETY: empty, or one word no kernel reads; all are replaced before use.
        unsafe {
            Ok(Self {
                requested: sieve_request().0,
                gather: sieve_request().1,
                count: check || flag("HSPZ_GATE_SIEVE_COUNT"),
                sub_mat: sub_mat.to_vec(),
                ref_has_x: false,
                ref_pk: uninitialized(stream, 0)?,
                qry_pk: [uninitialized(stream, 0)?, uninitialized(stream, 0)?],
                qry_bad: [uninitialized(stream, 0)?, uninitialized(stream, 0)?],
                args: None,
                classes: None,
                words: sieve_words(),
                masked: sieve_masked(),
                said: String::new(),
                cand_bits: uninitialized(stream, 1)?,
                guard: SieveGuard::new(sieve_guard_period()),
                host: check.then(SieveHost::default),
            })
        }
    }

    /// Frees the previous unit's query planes. The pass that read them has
    /// drained the stream, as it has for the query buffers `swap_query` replaces.
    fn release_query(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        self.args = None;
        // What the last launches of one query said about its hits says nothing
        // about the next query's: start it with a sample.
        self.guard = SieveGuard::new(self.guard.period);
        for strand in 0..2 {
            // SAFETY: empty buffers, never read.
            unsafe {
                self.qry_pk[strand] = uninitialized(stream, 0)?;
                self.qry_bad[strand] = uninitialized(stream, 0)?;
            }
        }
        Ok(())
    }
}

/// One thread per packed group, grid-stride.
#[cfg(feature = "gate-sieve")]
fn pack_launch(groups: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((groups as u32).div_ceil(SCAN_BLOCK).clamp(1, 1 << 14), 1, 1),
        block_dim: (SCAN_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Engine {
    /// Round 122: whether this engine's batches are seeded without transitions
    /// ([`FoldRequest`]). It can turn false once, when a chunk takes the
    /// folded tables' memory.
    pub fn folded(&self) -> bool {
        self.fold_alt.is_some()
    }

    /// Round 124: whether the folded batch about to be counted will be written
    /// in bucket order from its seeds, the tests of the chunk loop taken ahead
    /// of it (a folded batch is one chunk). Only then does nothing downstream
    /// need a count per seed, and `fold_seed_counts` can stand for
    /// `find_num_hits_lo`, its scan and `run_bucket_counts`. `HSPZ_FOLD_COUNT=0`
    /// keeps those.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn fused_count_now(&self) -> bool {
        fold_count_fused()
            && self.fold_active
            && self.sieve.args.is_some()
            && !self.sieve.guard.next_is_plain()
            && matches!(
                self.bucket_mode,
                BucketMode::Forced(true) | BucketMode::Decided(true)
            )
            && self.survivor_slots
            && self.bucket_match
            && !self.dump_raw
            && self.census.is_none()
            && !self.collect_hit_stats
            && order_check_limit() == 0
            && self.paired.is_none()
            && !self.buf_bucket_nibbles.is_empty()
            && self.hits_bucketed == HitsBucketed::On
    }

    #[cfg(not(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    )))]
    fn fused_count_now(&self) -> bool {
        false
    }

    /// Round 124: how many counts a fused batch scans, one per run and bucket.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn fused_scan_len(&self, num_seeds: u32) -> u32 {
        self.n_buckets * num_seeds.div_ceil(fold_run_len())
    }

    #[cfg(not(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    )))]
    fn fused_scan_len(&self, num_seeds: u32) -> u32 {
        num_seeds
    }

    /// Round 124: `fold_seed_counts` over the seeds of `slot`: the runs' counts
    /// per bucket, and each seed's word and start for the group kernel.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn launch_fold_seed_counts(&mut self, slot: usize, num_seeds: u32) -> Result<(), DriverError> {
        let run_len = fold_run_len();
        let num_runs = num_seeds.div_ceil(run_len);
        let row_shift = self.bounds_row_shift();
        // SAFETY: the kernel writes every count of a live run and every live
        // seed's word and start before anything reads them.
        unsafe {
            reserve(
                &mut self.buf_seed_counts,
                &self.stream,
                (self.n_buckets * num_runs) as usize,
                &mut self.pipeline_syncs,
            )?;
            reserve(
                &mut self.buf_seed_nib,
                &self.stream,
                num_seeds as usize,
                &mut self.pipeline_syncs,
            )?;
            reserve(
                &mut self.buf_seed_lo,
                &self.stream,
                num_seeds as usize,
                &mut self.pipeline_syncs,
            )?;
            self.module.fold_seed_counts(
                &self.stream,
                LaunchConfig {
                    grid_dim: (num_seeds.div_ceil(BLOCK_SIZE), 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.seed_slots[slot],
                &self.buf_bucket_bounds,
                num_seeds,
                self.n_buckets,
                row_shift,
                num_runs,
                run_len,
                &mut self.buf_seed_counts,
                &mut self.buf_seed_nib,
                &mut self.buf_seed_lo,
            )
        }
    }

    #[cfg(not(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    )))]
    fn launch_fold_seed_counts(
        &mut self,
        _slot: usize,
        _num_seeds: u32,
    ) -> Result<(), DriverError> {
        Ok(())
    }

    /// Round 124: the block scan of a fused batch's run counts, its sums into
    /// `scr_hit_sums` where the hit-count scan's go.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn launch_scan_sat(&mut self, cfg: LaunchConfig, len: u32) -> Result<(), DriverError> {
        // SAFETY: one element per thread over `len`, one sum slot per block.
        unsafe {
            self.module.scan_blocks_sat(
                &self.stream,
                cfg,
                &mut self.buf_seed_counts,
                &mut self.scr_hit_sums,
                len,
            )
        }
    }

    #[cfg(not(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    )))]
    fn launch_scan_sat(&mut self, _cfg: LaunchConfig, _len: u32) -> Result<(), DriverError> {
        Ok(())
    }

    /// Round 124: the last step of that scan, from the offsets the host sent.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn launch_add_run_offsets(&mut self, cfg: LaunchConfig, len: u32) -> Result<(), DriverError> {
        // SAFETY: the geometry of the scan, so block `b` reads offset `b`.
        unsafe {
            self.module.add_block_offsets(
                &self.stream,
                cfg,
                &mut self.buf_seed_counts,
                &self.scr_hit_offsets,
                len,
            )
        }
    }

    #[cfg(not(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    )))]
    fn launch_add_run_offsets(&mut self, _cfg: LaunchConfig, _len: u32) -> Result<(), DriverError> {
        Ok(())
    }

    /// Round 122: the next batch call is on plain seeds, all variants: a batch
    /// that came back `oversized` from a folded engine, sent again.
    pub fn plain_next(&mut self) {
        self.plain_once = true;
    }

    /// Makes the engine's tables the folded set or the plain one. Queued
    /// kernels hold device addresses, not these handles.
    fn use_tables(&mut self, fold: bool) {
        if self.fold_active == fold {
            return;
        }
        let Some(alt) = self.fold_alt.as_mut() else {
            return;
        };
        std::mem::swap(&mut self.index_table, &mut alt[0]);
        std::mem::swap(&mut self.pos_table, &mut alt[1]);
        #[cfg(all(
            feature = "find-hits-groups",
            feature = "ref-loc-buckets",
            feature = "gate-sieve"
        ))]
        {
            std::mem::swap(&mut self.buf_bucket_nibbles, &mut alt[2]);
            std::mem::swap(&mut self.buf_bucket_bounds, &mut alt[3]);
        }
        self.fold_active = fold;
    }

    /// Frees the folded tables; the engine is plain from here on.
    fn release_fold(&mut self, why: &str) {
        self.use_tables(false);
        if self.fold_alt.take().is_some() {
            fold_totals::RELEASED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if self.timing {
                eprintln!(
                    "  transition fold: dropped for this reference, {why} ({} of {} batches sent \
                     again, {} hits fit beside it)",
                    self.fold_again, self.fold_seen, self.fold_capacity
                );
            }
        }
    }
}

#[cfg(feature = "gate-sieve")]
impl Engine {
    /// Words per row of bucket edges, as a shift: the next power of two above
    /// `n_buckets` (a row holds `n_buckets + 1` edges), so a row of up to seven
    /// buckets is one 32-byte sector.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn bounds_row_shift(&self) -> u32 {
        (self.n_buckets + 1).next_power_of_two().trailing_zeros()
    }

    /// Round 122: builds the folded tables beside the plain ones
    /// ([`FoldRequest`]) and makes them the engine's: every k-mer's folded
    /// length (`fold_counts`), the scan, one pass that copies each variant's
    /// positions window by window (`fold_fill`), and the bucket tables of the
    /// result. A k-mer's list is in window order, which is all the bucket
    /// tables and the group kernel need; inside a window it is variant by
    /// variant, not ascending.
    ///
    /// Not built, with a line under `--time`, when the bucket order is not
    /// written from the seeds for this reference, the folded positions do not
    /// fit 32-bit offsets, or `max_hits` hits no longer fit beside them.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn fold_tables(
        &mut self,
        req: FoldRequest,
        host_index: &[u32],
        host_pos: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_kmers = self.index_table.len() as u32;
        let kmer_size = n_kmers.trailing_zeros() / 2;
        let plain = self.pos_table.len() as u64;
        let total = plain * u64::from(req.mask.count_ones() + 1);
        // Folded positions, their ends, the counts per k-mer and at most a row
        // of edges for every k-mer.
        let bytes = 4 * total
            + 8 * u64::from(n_kmers)
            + ((4 * u64::from(n_kmers)) << self.bounds_row_shift());
        let left =
            u64::from(self.hit_capacity).saturating_sub(bytes.div_ceil(req.bytes_per_hit.max(1)));
        let why_not = if req.mask == 0 {
            Some("the seed has no transition".to_string())
        } else if fold_totals::CUT.load(std::sync::atomic::Ordering::Relaxed) {
            Some(
                "an earlier reference of this run dropped it, its batches being cut at --max-hits"
                    .to_string(),
            )
        } else if self.census.is_some() {
            Some("the census counts by plain seed".to_string())
        } else if self.buf_bucket_nibbles.is_empty() {
            Some("the bucket order is not written from the seeds".to_string())
        } else if total > u64::from(u32::MAX) {
            Some(format!("{total} folded positions do not fit 32 bits"))
        } else if left < u64::from(self.max_hits) {
            Some(format!(
                "{:.1} GiB of tables leave room for {left} hits, below --max-hits {}",
                bytes as f64 / (1u64 << 30) as f64,
                self.max_hits
            ))
        } else if (device_memory().0 as u64) < bytes + (1 << 30) {
            Some(format!(
                "{:.1} GiB of tables, {:.1} GiB free on the device",
                bytes as f64 / (1u64 << 30) as f64,
                device_memory().0 as f64 / (1u64 << 30) as f64
            ))
        } else {
            None
        };
        if let Some(why) = why_not {
            if self.timing {
                eprintln!("  transition fold: off for this reference, {why}");
            }
            return Ok(());
        }
        let t = Instant::now();
        // SAFETY: `fold_counts` writes every word.
        let mut ends = unsafe { uninitialized::<u32>(&self.stream, n_kmers as usize)? };
        let sblocks = n_kmers.div_ceil(SCAN_BLOCK);
        let mut d_sums = DeviceBuffer::<u32>::zeroed(&self.stream, sblocks as usize)?;
        let scan = LaunchConfig {
            grid_dim: (sblocks, 1, 1),
            block_dim: (SCAN_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one thread per k-mer writing its own word, then the in-block
        // scan with its block totals.
        unsafe {
            self.module.fold_counts(
                &self.stream,
                elementwise(),
                &self.index_table,
                req.mask,
                kmer_size,
                n_kmers,
                &mut ends,
            )?;
            self.module
                .scan_blocks(&self.stream, scan, &mut ends, &mut d_sums, n_kmers)?;
        }
        // Once per reference, so the block totals take the host's way.
        let mut sums = d_sums.to_host_vec(&self.stream)?;
        let mut acc = 0u64;
        for sum in sums.iter_mut() {
            let here = *sum;
            *sum = acc as u32;
            acc += u64::from(here);
        }
        if acc != total {
            return Err(format!(
                "transition fold: {acc} folded positions counted, {total} expected"
            )
            .into());
        }
        let d_offsets = DeviceBuffer::from_host(&self.stream, &sums)?;
        // SAFETY: the scan's last step; then one thread per k-mer writes the
        // `ends[k - 1]..ends[k]` slots of the folded positions, all of them.
        // Round 124: the pass also writes the folded table's bucket tables, a
        // row of bucket starts for every k-mer and the word that names it.
        let row_shift = self.bounds_row_shift();
        let (folded, nibbles, bounds) = unsafe {
            self.module
                .add_block_offsets(&self.stream, scan, &mut ends, &d_offsets, n_kmers)?;
            let mut folded = uninitialized::<u32>(&self.stream, total as usize)?;
            let mut nibbles = uninitialized::<u32>(&self.stream, n_kmers as usize)?;
            let mut bounds = uninitialized::<u32>(&self.stream, (n_kmers as usize) << row_shift)?;
            self.module.fold_fill(
                &self.stream,
                elementwise(),
                &self.index_table,
                &self.pos_table,
                &ends,
                req.mask,
                kmer_size,
                self.seed_size,
                self.bucket_shift,
                self.n_buckets,
                row_shift,
                n_kmers,
                &mut folded,
                &mut bounds,
                &mut nibbles,
            )?;
            (folded, nibbles, bounds)
        };
        self.stream.synchronize()?;
        let built = t.elapsed();
        if req.check {
            self.check_fold(
                req.mask, kmer_size, host_index, host_pos, &ends, &folded, &bounds,
            )?;
        }
        self.fold_alt = Some(Box::new([ends, folded, nibbles, bounds]));
        self.use_tables(true);
        self.fold_capacity = left.min(u64::from(u32::MAX)) as u32;
        // A test's way to the release: `HSPZ_FOLD_CAPACITY` lowers the figure.
        if let Some(cap) = std::env::var("HSPZ_FOLD_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            self.fold_capacity = self.fold_capacity.min(cap);
        }
        fold_totals::BINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.timing {
            eprintln!(
                "  transition fold: {plain} positions -> {total} ({:.1} GiB) and {} MiB of bucket rows, \
                 built in {:.1} ms; chunks up to {} hits beside it{}",
                total as f64 * 4.0 / (1u64 << 30) as f64,
                ((n_kmers as usize) << row_shift) >> 18,
                built.as_secs_f64() * 1e3,
                self.fold_capacity,
                if req.check {
                    " (checked against the host's)"
                } else {
                    ""
                },
            );
        }
        Ok(())
    }

    /// `HSPZ_FOLD=check`: the device's folded tables against
    /// `seed::fold_tables` on the host: the same ends, and for every k-mer the
    /// same positions, in window order.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn check_fold(
        &mut self,
        mask: u32,
        kmer_size: u32,
        host_index: &[u32],
        host_pos: &[u32],
        ends: &DeviceBuffer<u32>,
        folded: &DeviceBuffer<u32>,
        bounds: &DeviceBuffer<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let (want_ends, want_pos) =
            crate::seed::fold_tables(host_index, host_pos, mask, kmer_size as usize, threads)
                .ok_or("transition fold check: the host's table does not fit 32 bits")?;
        let got_ends = ends.to_host_vec(&self.stream)?;
        if got_ends != want_ends {
            let k = got_ends.iter().zip(&want_ends).position(|(a, b)| a != b);
            return Err(
                format!("transition fold check: the ends differ, first at k-mer {k:?}").into(),
            );
        }
        let got_pos = folded.to_host_vec(&self.stream)?;
        let got_rows = bounds.to_host_vec(&self.stream)?;
        let row_shift = self.bounds_row_shift();
        let (shift, top, seed_size) = (self.bucket_shift, self.n_buckets - 1, self.seed_size);
        let bucket = |p: u32| ((p + seed_size) >> shift).min(top);
        let n = want_ends.len();
        let per = n.div_ceil(threads);
        let bad = std::sync::atomic::AtomicUsize::new(usize::MAX);
        std::thread::scope(|scope| {
            for c in 0..threads {
                let (got_pos, got_rows, want_pos, want_ends, bad, bucket) =
                    (&got_pos, &got_rows, &want_pos, &want_ends, &bad, &bucket);
                scope.spawn(move || {
                    let mut list = Vec::new();
                    for k in (c * per)..((c + 1) * per).min(n) {
                        let lo = if k > 0 { want_ends[k - 1] as usize } else { 0 };
                        let hi = want_ends[k] as usize;
                        let got = &got_pos[lo..hi];
                        let in_order = got.windows(2).all(|w| bucket(w[0]) <= bucket(w[1]));
                        // Word `w` of the k-mer's row: where its bucket `w`
                        // starts, the list's end past the last bucket.
                        let row = &got_rows[k << row_shift..(k + 1) << row_shift];
                        let rows_ok = row.iter().enumerate().all(|(w, &at)| {
                            at as usize == lo + got.partition_point(|&p| bucket(p) < w as u32)
                        });
                        list.clear();
                        list.extend_from_slice(got);
                        list.sort_unstable();
                        if !in_order || !rows_ok || list != want_pos[lo..hi] {
                            bad.fetch_min(k, std::sync::atomic::Ordering::Relaxed);
                            return;
                        }
                    }
                });
            }
        });
        match bad.into_inner() {
            usize::MAX => Ok(()),
            k => Err(format!(
                "transition fold check: k-mer {k} holds other positions, is out of window order or has a wrong row"
            )
            .into()),
        }
    }

    /// Round 119: the two tables the group line reads, once per reference: every
    /// k-mer's hits per bucket in four bits (`bucket_nibbles`, the size of the
    /// index table), and a row of bucket edges for each k-mer with a count
    /// above 14 (`bucket_overflow_rows`).
    ///
    /// Left empty, which keeps the count and the scatter, when
    /// `HSPZ_HITS_BUCKETED=0`, the window is not the planes' one, the run is
    /// `--paired` (a pair takes a slice of a k-mer's positions; `set_pairs`
    /// drops the tables, because the pairs arrive after the engine is built),
    /// the group kernels are off or under their check (which compares the raw
    /// anchors), there are more than eight buckets, the seed has more than 12
    /// care bases (the counts would be a gibibyte, far past the L2, and that
    /// case is not measured), or the rows would
    /// take more than the estimate allows them (`plan::unit_device_bytes`: a
    /// quarter of a byte per reference base, at least 64 MiB).
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn build_bucket_tables(&mut self) -> Result<(), DriverError> {
        if self.hits_bucketed == HitsBucketed::Off
            || !self.bucket_planes
            || self.paired.is_some()
            || self.find_hits_groups != FindHitsGroups::On
            || self.n_buckets == 0
            || self.n_buckets > 8
            || self.index_table.len() > 1 << 24
        {
            return Ok(());
        }
        let n_kmers = self.index_table.len() as u32;
        let row_shift = self.bounds_row_shift();
        let t = Instant::now();
        // SAFETY: `bucket_nibbles` writes every word of both buffers.
        let (mut nibbles, mut sat) = unsafe {
            (
                uninitialized::<u32>(&self.stream, n_kmers as usize)?,
                uninitialized::<u32>(&self.stream, n_kmers as usize)?,
            )
        };
        let sblocks = n_kmers.div_ceil(SCAN_BLOCK);
        let mut d_sums = DeviceBuffer::<u32>::zeroed(&self.stream, sblocks as usize)?;
        let scan = LaunchConfig {
            grid_dim: (sblocks, 1, 1),
            block_dim: (SCAN_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one thread per k-mer, each writing its own two words; then the
        // in-block scan of `sat` with its block totals.
        unsafe {
            self.module.bucket_nibbles(
                &self.stream,
                elementwise(),
                &self.index_table,
                &self.pos_table,
                self.seed_size,
                self.bucket_shift,
                self.n_buckets,
                n_kmers,
                &mut nibbles,
                &mut sat,
            )?;
            self.module
                .scan_blocks(&self.stream, scan, &mut sat, &mut d_sums, n_kmers)?;
        }
        // Once per reference, so the block totals take the host's way.
        let mut sums = d_sums.to_host_vec(&self.stream)?;
        let mut rows = 0u32;
        for sum in sums.iter_mut() {
            let here = *sum;
            *sum = rows;
            rows += here;
        }
        let row_bytes = (rows as usize * 4) << row_shift;
        // Most k-mers of the folded table have a row: its memory is counted
        // with the fold's ([`fold_tables`](Self::fold_tables)).
        let allowed = if self.fold_active {
            usize::MAX
        } else {
            (self.ref_len as usize / 4).max(64 << 20)
        };
        if !bucket_rows_fit(rows) || row_bytes > allowed {
            if self.timing {
                eprintln!(
                    "  bucket order from the seeds: off for this reference, {rows} k-mers need a row \
                     of edges ({} MiB, {} allowed)",
                    row_bytes >> 20,
                    allowed >> 20
                );
            }
            return Ok(());
        }
        if rows > 0 {
            let d_offsets = DeviceBuffer::from_host(&self.stream, &sums)?;
            // SAFETY: the scan's last step, then one thread per k-mer: a
            // saturated one writes its own row, which `rows` counted, and its
            // own word.
            unsafe {
                self.module
                    .add_block_offsets(&self.stream, scan, &mut sat, &d_offsets, n_kmers)?;
                self.buf_bucket_bounds = uninitialized(&self.stream, (rows as usize) << row_shift)?;
                self.module.bucket_overflow_rows(
                    &self.stream,
                    elementwise(),
                    &self.index_table,
                    &self.pos_table,
                    self.seed_size,
                    self.bucket_shift,
                    self.n_buckets,
                    row_shift,
                    n_kmers,
                    &sat,
                    &mut nibbles,
                    &mut self.buf_bucket_bounds,
                )?;
            }
            self.stream.synchronize()?;
        }
        self.buf_bucket_nibbles = nibbles;
        if self.timing {
            eprintln!(
                "  bucket order from the seeds: {} MiB of counts per k-mer, {rows} k-mers with a row of \
                 edges ({:.1} MiB), built in {:.1} ms{}",
                (n_kmers as usize * 4) >> 20,
                row_bytes as f64 / (1 << 20) as f64,
                t.elapsed().as_secs_f64() * 1e3,
                if matches!(self.hits_bucketed, HitsBucketed::Check(_)) {
                    " (check)"
                } else {
                    ""
                },
            );
        }
        Ok(())
    }

    /// Round 119: one chunk's bucket order written from its seeds
    /// into `buf_sorted_anchor`.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn launch_hits_bucketed(
        &mut self,
        slot: usize,
        start_seed_index: u32,
        num_seeds: u32,
        counted: bool,
    ) -> Result<(), DriverError> {
        // One warp of the two kernels below per run of `run_len` seeds: 32,
        // or fewer of the folded table's, which have 13 times the hits each.
        let run_len = if self.fold_active { fold_run_len() } else { 32 };
        let num_runs = num_seeds.div_ceil(run_len);
        let counts_len = self.n_buckets * num_runs;
        let row_shift = self.bounds_row_shift();
        let by_runs = LaunchConfig {
            grid_dim: ((num_runs * 32).div_ceil(BLOCK_SIZE), 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        // Round 124: a batch counted by `fold_seed_counts` arrives with its
        // runs' counts scanned and its seeds' words written.
        if !counted {
            // SAFETY: `run_bucket_counts` writes every count before the scan reads
            // it, and every live seed's word before the group kernel reads it.
            unsafe {
                reserve(
                    &mut self.buf_seed_counts,
                    &self.stream,
                    counts_len as usize,
                    &mut self.pipeline_syncs,
                )?;
                reserve(
                    &mut self.buf_seed_nib,
                    &self.stream,
                    num_seeds as usize,
                    &mut self.pipeline_syncs,
                )?;
            }
            let stage = self.stage();
            // SAFETY: one thread per seed slot; lane 0 of run `r` writes the
            // `n_buckets` slots `b * num_runs + r`.
            unsafe {
                self.module.run_bucket_counts(
                    &self.stream,
                    by_runs,
                    &self.seed_slots[slot],
                    &self.buf_bucket_nibbles,
                    &self.buf_bucket_bounds,
                    start_seed_index,
                    num_seeds,
                    self.n_buckets,
                    row_shift,
                    num_runs,
                    run_len,
                    &mut self.buf_seed_counts,
                    &mut self.buf_seed_nib,
                )?;
            }
            self.end_stage(stage, "run_bucket_counts")?;

            let sblocks = counts_len.div_ceil(SCAN_BLOCK);
            // SAFETY: `scan_blocks` writes one sum for every launched block.
            unsafe {
                reserve(
                    &mut self.scr_run_sums,
                    &self.stream,
                    sblocks as usize,
                    &mut self.pipeline_syncs,
                )?;
            }
            let scan = LaunchConfig {
                grid_dim: (sblocks, 1, 1),
                block_dim: (SCAN_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            let stage = self.stage();
            // SAFETY: the three launches of the bucket pass's own scan, over the
            // runs' histogram instead of the blocks'.
            unsafe {
                self.module.scan_blocks(
                    &self.stream,
                    scan,
                    &mut self.buf_seed_counts,
                    &mut self.scr_run_sums,
                    counts_len,
                )?;
                self.module.scan_exclusive_one_block(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (kernels::ONE_BLOCK_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut self.scr_run_sums,
                    sblocks,
                )?;
                self.module.add_block_offsets(
                    &self.stream,
                    scan,
                    &mut self.buf_seed_counts,
                    &self.scr_run_sums,
                    counts_len,
                )?;
            }
            self.end_stage(stage, "run_bucket_scan")?;
        }

        let stage = self.stage();
        // SAFETY: one thread per seed slot, as `find_hits_groups`; the scanned
        // counts and the table's rows give every hit of the chunk its own slot
        // in `0..iter_num_hits`.
        unsafe {
            self.module.find_hits_groups_bucketed(
                &self.stream,
                by_runs,
                &self.pos_table,
                &self.buf_bucket_bounds,
                &self.buf_seed_nib,
                &self.seed_slots[slot],
                self.seed_size,
                &self.buf_seed_lo,
                &self.buf_seed_counts,
                self.n_buckets,
                row_shift,
                num_runs,
                run_len,
                start_seed_index,
                num_seeds,
                &mut self.buf_sorted_anchor,
            )?;
        }
        self.end_stage(stage, "find_hits_bucketed")?;
        Ok(())
    }

    /// Round 119, `HSPZ_HITS_BUCKETED=check`: the count and the scatter have
    /// just written this chunk's bucket order; write it again from the seeds
    /// into a scratch buffer and compare the two on the host. The order inside
    /// a bucket differs by design, so each bucket's anchors are compared as a
    /// sorted list: the same multiset in the same bucket.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    fn check_hits_bucketed(
        &mut self,
        slot: usize,
        start_seed_index: u32,
        num_seeds: u32,
        num_hits: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::atomic::Ordering::Relaxed;
        let n = num_hits as usize;
        self.sync_pipeline()?;
        let mut want = copy_prefix(&self.buf_sorted_anchor, &self.stream, n)?;
        // Zeroed scratch: a slot the seed-driven kernel never wrote cannot pass
        // on what the scatter left there.
        let mut scratch = DeviceBuffer::<u64>::zeroed(&self.stream, n)?;
        std::mem::swap(&mut self.buf_sorted_anchor, &mut scratch);
        let launched = self.launch_hits_bucketed(slot, start_seed_index, num_seeds, false);
        let synced = self.sync_pipeline();
        std::mem::swap(&mut self.buf_sorted_anchor, &mut scratch);
        launched?;
        synced?;
        let mut got = copy_prefix(&scratch, &self.stream, n)?;
        let (shift, top) = (self.bucket_shift, self.n_buckets - 1);
        let bucket_of = |a: u64| ((a as u32) >> shift).min(top);
        let mut lo = 0usize;
        while lo < n {
            let b = bucket_of(want[lo]);
            let mut hi = lo + 1;
            while hi < n && bucket_of(want[hi]) == b {
                hi += 1;
            }
            if let Some(p) = (lo..hi).find(|&p| bucket_of(got[p]) != b) {
                return Err(format!(
                    "hits-bucketed check: slot {p} of {n} (chunk seeds {start_seed_index}+{num_seeds}) \
                     holds anchor {:#018x} of bucket {}, in the range of bucket {b}",
                    got[p],
                    bucket_of(got[p])
                )
                .into());
            }
            want[lo..hi].sort_unstable();
            got[lo..hi].sort_unstable();
            if let Some(p) = (lo..hi).find(|&p| got[p] != want[p]) {
                return Err(format!(
                    "hits-bucketed check: bucket {b} of a chunk of {n} hits (seeds \
                     {start_seed_index}+{num_seeds}) differs at sorted rank {}: {:#018x} from the \
                     seeds, {:#018x} from the scatter",
                    p - lo,
                    got[p],
                    want[p]
                )
                .into());
            }
            lo = hi;
        }
        // How many of the chunk's seeds had a saturated count and read a row.
        let words = copy_prefix(&self.buf_seed_nib, &self.stream, num_seeds as usize)?;
        let saturated = words
            .iter()
            .filter(|&&w| w & (w >> 1) & (w >> 2) & (w >> 3) & 0x1111_1111 != 0)
            .count();
        hits_bucketed_totals::CHECKED_SEEDS.fetch_add(u64::from(num_seeds), Relaxed);
        hits_bucketed_totals::SATURATED_SEEDS.fetch_add(saturated as u64, Relaxed);
        hits_bucketed_totals::CHECKED_CHUNKS.fetch_add(1, Relaxed);
        hits_bucketed_totals::CHECKED_HITS.fetch_add(n as u64, Relaxed);
        Ok(())
    }

    /// Packs the resident reference into its planes, on the device.
    fn sieve_pack_reference(&mut self, ref_enc: &[u8]) -> Result<(), DriverError> {
        if !self.sieve.requested || ref_enc.is_empty() {
            return Ok(());
        }
        self.sieve.ref_has_x = memchr::memchr(crate::sequence::X_NT, ref_enc).is_some();
        let groups = sieve::groups(ref_enc.len());
        // SAFETY: `pack_planes` writes every group before the gate reads one; the
        // reference has no bad plane, so that argument is never written.
        unsafe {
            self.sieve.ref_pk = uninitialized(&self.stream, groups)?;
            self.module.pack_planes(
                &self.stream,
                pack_launch(groups),
                &self.ref_seq,
                self.ref_len,
                groups as u32,
                sieve::REF_FILL,
                0,
                &mut self.sieve.ref_pk,
                &mut self.sieve.cand_bits,
            )?;
        }
        if let Some(host) = self.sieve.host.as_mut() {
            host.ref_enc = ref_enc.to_vec();
            host.ref_pk = sieve::pack(ref_enc, sieve::REF_FILL).0;
            let device = copy_prefix(&self.sieve.ref_pk, &self.stream, groups)?;
            let wrong = device
                .iter()
                .zip(&host.ref_pk)
                .filter(|(d, h)| d != h)
                .count();
            sieve_totals::PLANE_MISMATCHES
                .fetch_add(wrong as u64, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// Decides whether the bound holds for this unit and packs both query strands.
    fn sieve_pack_query(&mut self, query: &[u8], query_rc: &[u8]) -> Result<(), DriverError> {
        if !self.sieve.requested {
            return Ok(());
        }
        let x = crate::sequence::X_NT;
        let qry_has_x = memchr::memchr(x, query).is_some() || memchr::memchr(x, query_rc).is_some();
        // Plane positions are 32-bit and carry four groups of padding on each side.
        let too_long = |len: usize| len > (u32::MAX - 64 * sieve::PADG) as usize;
        let decision = if query.is_empty() || self.sieve.ref_pk.is_empty() {
            Err("an empty sequence".to_string())
        } else if too_long(self.ref_len as usize) || too_long(query.len()) {
            Err("a sequence too long for 32-bit plane positions".to_string())
        } else {
            sieve::classes(
                &self.sieve.sub_mat,
                self.xdrop,
                self.sieve.ref_has_x,
                qry_has_x,
                self.sieve.masked,
            )
        };
        let state = match &decision {
            Ok(c) => format!(
                "on — match {}, transition {}, transversion {}; {} right + {} left bases{}{}{}",
                c.um,
                c.uti,
                c.utv,
                sieve::RIGHT,
                sieve::LEFT,
                if self.sieve.masked {
                    ""
                } else {
                    ", query without a mask plane"
                },
                if self.sieve.words == (sieve::RIGHT_WORDS, sieve::LEFT_WORDS) {
                    String::new()
                } else {
                    format!(
                        ", an undecided side carried to {} right, {} left",
                        32 * self.sieve.words.0,
                        32 * self.sieve.words.1
                    )
                },
                match self.sieve.gather {
                    Some(true) => "; gathered reference",
                    Some(false) => "",
                    None => "; reference gathered on the plain gate only",
                }
            ),
            Err(why) => format!("off — {why}"),
        };
        if self.timing && state != self.sieve.said {
            eprintln!("  gate sieve: {state}");
            self.sieve.said = state;
        }
        self.sieve.classes = decision.ok();
        self.sieve.args = self.sieve.classes.map(|c| c.kernel_args());
        if self.sieve.args.is_none() {
            return Ok(());
        }

        let fill = if self.sieve.masked {
            sieve::REF_FILL
        } else {
            sieve::QRY_FILL
        };
        for (strand, enc) in [query, query_rc].into_iter().enumerate() {
            // `--paired` swaps in the plus strand alone; a strand that is not
            // there is never gated, and its planes stay empty.
            if enc.is_empty() {
                continue;
            }
            let groups = sieve::groups(enc.len());
            let seq = if strand == 0 {
                &self.query_seq
            } else {
                &self.query_rc_seq
            };
            // Without the mask plane the kernels never read it: one word that
            // nothing writes stands in for it.
            let masked = self.sieve.masked;
            let bad_len = if masked { groups } else { 1 };
            // SAFETY: `pack_planes` writes every group of the planes it is asked
            // for before the gate reads one.
            unsafe {
                self.sieve.qry_pk[strand] = uninitialized(&self.stream, groups)?;
                self.sieve.qry_bad[strand] = uninitialized(&self.stream, bad_len)?;
                self.module.pack_planes(
                    &self.stream,
                    pack_launch(groups),
                    seq,
                    enc.len() as u32,
                    groups as u32,
                    fill,
                    masked as u32,
                    &mut self.sieve.qry_pk[strand],
                    &mut self.sieve.qry_bad[strand],
                )?;
            }
            if let Some(host) = self.sieve.host.as_mut() {
                let (pk, bad) = sieve::pack(enc, fill);
                let d_pk = copy_prefix(&self.sieve.qry_pk[strand], &self.stream, groups)?;
                let mut wrong = d_pk.iter().zip(&pk).filter(|(d, h)| d != h).count();
                if masked {
                    let d_bad = copy_prefix(&self.sieve.qry_bad[strand], &self.stream, groups)?;
                    wrong += d_bad.iter().zip(&bad).filter(|(d, h)| d != h).count();
                }
                sieve_totals::PLANE_MISMATCHES
                    .fetch_add(wrong as u64, std::sync::atomic::Ordering::Relaxed);
                host.qry_enc[strand] = enc.to_vec();
                host.qry_pk[strand] = pk;
                host.qry_bad[strand] = bad;
            }
        }
        Ok(())
    }

    /// The guard's sample of one gate launch: how many of its `n` hits the bound
    /// passed to the exact gate. Block sums on the device, one small copy.
    fn sieve_count_candidates(&mut self, n: u32) -> Result<u64, DriverError> {
        let words = n.div_ceil(32);
        let blocks = words.div_ceil(SCAN_BLOCK);
        // SAFETY: `count_bits` writes one sum per launched block before the copy.
        // The survivor count, which shares the buffer, runs after this returns.
        unsafe {
            reserve(
                &mut self.scr_surv_sums,
                &self.stream,
                blocks as usize,
                &mut self.pipeline_syncs,
            )?;
            self.module.count_bits(
                &self.stream,
                LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (SCAN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.sieve.cand_bits,
                &mut self.scr_surv_sums,
                words,
            )?;
        }
        let sums = copy_prefix(&self.scr_surv_sums, &self.stream, blocks as usize)?;
        self.absorb_sync()?;
        Ok(sums.iter().map(|&c| u64::from(c)).sum())
    }

    /// The count and check diagnostics for one gate launch: reads the candidate
    /// words back and, under `check`, replays every hit on the host. Returns the
    /// candidate count.
    fn sieve_audit(&mut self, rev: bool, n: u32, bucket_pass: bool) -> Result<u64, DriverError> {
        use std::sync::atomic::Ordering::Relaxed;
        let words = n.div_ceil(32) as usize;
        let cand = copy_prefix(&self.sieve.cand_bits, &self.stream, words)?;
        self.absorb_sync()?;
        let candidates: u64 = cand.iter().map(|w| u64::from(w.count_ones())).sum();
        sieve_totals::HITS.fetch_add(u64::from(n), Relaxed);
        sieve_totals::CANDIDATES.fetch_add(candidates, Relaxed);
        let (Some(host), Some(classes)) = (self.sieve.host.as_ref(), self.sieve.classes) else {
            return Ok(candidates);
        };

        // The anchors the gate read and the keep bits it wrote, in the gate's
        // order: sorted slots under the bucket pass, raw hits otherwise.
        let anchors = if bucket_pass {
            copy_prefix(&self.buf_sorted_anchor, &self.stream, n as usize)?
        } else {
            copy_prefix(&self.buf_anchor, &self.stream, n as usize)?
        };
        let keep_bits = copy_prefix(&self.buf_flags_bits, &self.stream, words)?;
        let bit = |bits: &[u32], i: usize| bits[i / 32] >> (i % 32) & 1 != 0;
        let device_keep = |i: usize| bit(&keep_bits, i);

        let strand = rev as usize;
        let (xdrop, t_min, sub_mat) = (self.xdrop, self.gate_thresh, &self.sieve.sub_mat);
        let carried = self.sieve.words;
        let masked = self.sieve.masked;
        let threads = std::thread::available_parallelism()
            .map_or(1, |t| t.get())
            .min(16);
        let per = (n as usize).div_ceil(threads).max(1);
        let tallies: Vec<[u64; 4]> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..n as usize)
                .step_by(per)
                .map(|start| {
                    let (anchors, cand, device_keep) = (&anchors, &cand, &device_keep);
                    scope.spawn(move || {
                        // kept, candidate mismatches, flag mismatches, kept but rejected
                        let mut t = [0u64; 4];
                        let end = (start + per).min(n as usize);
                        for (i, &anchor) in anchors.iter().enumerate().take(end).skip(start) {
                            let (rl, ql) = (anchor as u32, (anchor >> 32) as u32);
                            let model_cand = sieve::candidate(
                                &host.ref_pk,
                                &host.qry_pk[strand],
                                &host.qry_bad[strand],
                                rl,
                                ql,
                                &classes,
                                xdrop,
                                t_min,
                                carried,
                                masked,
                            );
                            let (right, left) = sieve::exact_sides(
                                &host.ref_enc,
                                &host.qry_enc[strand],
                                rl,
                                ql,
                                sub_mat,
                                xdrop,
                            );
                            let keep = right.wrapping_add(left) >= t_min;
                            let device_cand = bit(cand, i);
                            t[0] += keep as u64;
                            t[1] += (model_cand != device_cand) as u64;
                            t[2] += (keep != device_keep(i)) as u64;
                            t[3] += (keep && !device_cand) as u64;
                        }
                        t
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|w| w.join().expect("sieve check worker"))
                .collect()
        });
        let sum = |k: usize| tallies.iter().map(|t| t[k]).sum::<u64>();
        sieve_totals::CHECKED.fetch_add(u64::from(n), Relaxed);
        sieve_totals::KEPT.fetch_add(sum(0), Relaxed);
        sieve_totals::CANDIDATE_MISMATCHES.fetch_add(sum(1), Relaxed);
        sieve_totals::FLAG_MISMATCHES.fetch_add(sum(2), Relaxed);
        sieve_totals::KEPT_BUT_REJECTED.fetch_add(sum(3), Relaxed);
        Ok(candidates)
    }
}

/// Round 116: in which order a bucketed chunk's survivors reach `find_hsps`.
///
/// The bucket pass sorts a chunk's hits by reference window. Until round 116
/// the scatter also wrote every hit's raw id, the survivors were mapped back
/// through it and sorted, and `find_hsps` read `buf_anchor` by raw id: the
/// order of before the pass, kept because it was the order of before. Nothing
/// that reaches the output depends on it: `find_hsps` extends each survivor on
/// its own, and the host orders the chunk's HSPs with `hsp::by_diagonal`, which
/// compares all four fields of a segment, so two segments it calls equal are
/// the same sixteen bytes and the sorted list is the same whatever order they
/// arrived in. So the survivors are taken by sorted slot, `find_hsps` reads the
/// sorted anchors, and the ids are neither written nor sorted.
///
/// `HSPZ_SURVIVOR_ORDER=raw` restores the ids. The paths that show an order
/// (`dump_raw`, the census, `HSPZ_BUCKET_ORDER_CHECK`) and the scatter of
/// before round 114 keep them too.
#[cfg(all(feature = "ref-loc-buckets", feature = "gate-sieve"))]
fn survivor_slots_request() -> bool {
    static REQUEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| match std::env::var("HSPZ_SURVIVOR_ORDER").as_deref() {
        Err(_) | Ok("slots") => true,
        Ok("raw") => false,
        Ok(other) => {
            eprintln!("warning: HSPZ_SURVIVOR_ORDER={other:?} is not slots or raw; using slots");
            true
        }
    })
}

/// Round 114: how the bucket kernels learn which lanes of a warp share a
/// bucket. `HSPZ_BUCKET_VOTES` unset or `match`: one `match.any`
/// (`bucket_count_match`, `bucket_scatter_match`). `loop`: one vote per bucket,
/// the kernels of rounds 83 to 113. Both write the same buffers
/// (`HSPZ_BUCKET_ORDER_CHECK` proves the permutation on the device).
#[cfg(feature = "ref-loc-buckets")]
fn bucket_match_request() -> bool {
    static REQUEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| match std::env::var("HSPZ_BUCKET_VOTES").as_deref() {
        Err(_) | Ok("match") => true,
        Ok("loop") => false,
        Ok(other) => {
            eprintln!("warning: HSPZ_BUCKET_VOTES={other:?} is not match or loop; using match");
            true
        }
    })
}

/// Round 110 (`find-hits-groups`): which kernel expands a chunk's seeds.
/// `HSPZ_FIND_HITS_GROUPS` unset: runs of 32 seeds per warp
/// (`find_hits_groups`). `0`: one warp or one thread per seed, as before round
/// 110. `check`: both, refusing the run at the first anchor that differs.
#[cfg(feature = "find-hits-groups")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FindHitsGroups {
    Off,
    On,
    Check,
}

#[cfg(feature = "find-hits-groups")]
fn find_hits_groups_request() -> FindHitsGroups {
    // Read once, so a value that is not understood is reported once and not by
    // every engine.
    static REQUEST: std::sync::OnceLock<FindHitsGroups> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| match std::env::var("HSPZ_FIND_HITS_GROUPS").as_deref() {
        Err(_) | Ok("1") => FindHitsGroups::On,
        Ok("0") => FindHitsGroups::Off,
        Ok("check") => FindHitsGroups::Check,
        Ok(other) => {
            eprintln!(
                "warning: HSPZ_FIND_HITS_GROUPS={other:?} is not 0, 1 or check; using 1 (runs of \
                     32 seeds per warp)"
            );
            FindHitsGroups::On
        }
    })
}

/// Chunks and hits the group kernel expanded, and those whose anchors the two
/// kernels agreed on.
#[cfg(feature = "find-hits-groups")]
pub mod find_hits_groups_totals {
    use std::sync::atomic::AtomicU64;
    pub static CHUNKS: AtomicU64 = AtomicU64::new(0);
    pub static HITS: AtomicU64 = AtomicU64::new(0);
    pub static CHECKED_CHUNKS: AtomicU64 = AtomicU64::new(0);
    pub static CHECKED_HITS: AtomicU64 = AtomicU64::new(0);
}

/// Round 119: how a bucketed chunk's anchors get into bucket order.
///
/// Until this round the hits were written in raw order (`find_hits`), counted
/// per bucket (`bucket_count`) and scattered: 32 bytes moved per hit after the
/// position was read. But a k-mer's positions ascend, so its hits of one bucket
/// are one contiguous slice of `pos_table`, and how many each bucket gets is a
/// property of the k-mer. With those counts in a table (`build_bucket_tables`),
/// a pass over the seeds gives every run of 32 seeds its place in every bucket
/// (`run_bucket_counts` and a scan), and `find_hits_groups_bucketed` writes each
/// anchor once, where it belongs. No raw anchor is written, nothing is counted
/// from anchors and nothing is scattered. The order inside a bucket is not the
/// raw order, which survivors by sorted slot (round 116) made free.
///
/// `HSPZ_HITS_BUCKETED` unset or `1`: on wherever it applies (a chunk whose
/// survivors go by sorted slot, a reference with the tables). `0`: the count
/// and the scatter. `check`, or `check:N` for the first N chunks, runs the
/// count and the scatter, then this path into a scratch buffer, and compares
/// the two bucket by bucket on the host (each bucket as a sorted list).
#[cfg(all(
    feature = "find-hits-groups",
    feature = "ref-loc-buckets",
    feature = "gate-sieve"
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HitsBucketed {
    Off,
    On,
    Check(u32),
}

#[cfg(all(
    feature = "find-hits-groups",
    feature = "ref-loc-buckets",
    feature = "gate-sieve"
))]
fn hits_bucketed_request() -> HitsBucketed {
    static REQUEST: std::sync::OnceLock<HitsBucketed> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| {
        let Ok(text) = std::env::var("HSPZ_HITS_BUCKETED") else {
            return HitsBucketed::On;
        };
        match text.as_str() {
            "0" => HitsBucketed::Off,
            "1" => HitsBucketed::On,
            "check" => HitsBucketed::Check(u32::MAX),
            other => match other.strip_prefix("check:").and_then(|n| n.parse().ok()) {
                Some(n) => HitsBucketed::Check(n),
                None => {
                    eprintln!(
                        "warning: HSPZ_HITS_BUCKETED={other:?} is not 0, 1, check or check:N; using 1"
                    );
                    HitsBucketed::On
                }
            },
        }
    })
}

/// Chunks and hits whose bucket order was written from the seeds, and those
/// compared with the scatter's.
#[cfg(all(
    feature = "find-hits-groups",
    feature = "ref-loc-buckets",
    feature = "gate-sieve"
))]
pub mod hits_bucketed_totals {
    use std::sync::atomic::AtomicU64;
    pub static CHUNKS: AtomicU64 = AtomicU64::new(0);
    pub static HITS: AtomicU64 = AtomicU64::new(0);
    pub static CHECKED_CHUNKS: AtomicU64 = AtomicU64::new(0);
    pub static CHECKED_HITS: AtomicU64 = AtomicU64::new(0);
    pub static CHECKED_SEEDS: AtomicU64 = AtomicU64::new(0);
    pub static SATURATED_SEEDS: AtomicU64 = AtomicU64::new(0);
}

/// `MAX_BLOCKS x MAX_THREADS`, the geometry KegAlign uses for its two
/// grid-stride elementwise kernels.
fn elementwise() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (MAX_BLOCKS, 1, 1),
        block_dim: (MAX_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Exclusive scan of one seed batch's per-block hit totals, in place, returning
/// the batch total.
///
/// The device scan, the chunk walk and every hit index are `u32`, as in
/// KegAlign. A batch that expands to more than `u32::MAX` hits (reachable with
/// unmasked, close-species inputs and bins of about 1 Gbp) used to wrap here in
/// release builds: a wrapped total above the cap aborted with a misleading
/// "not monotonic" error, and one below it took the single-chunk path, which
/// has no monotonicity check, and let `find_hits` write out of bounds. The sum
/// is taken in `u64` first, so such a batch is refused by name; every batch
/// that fits scans exactly as before.
fn exclusive_scan_hit_totals(sums: &mut [u32]) -> Result<u32, String> {
    let total: u64 = sums.iter().map(|&s| u64::from(s)).sum();
    let Ok(total) = u32::try_from(total) else {
        return Err(format!(
            "seed batch expands to {total} seed hits, more than the u32 hit index can \
             address ({}; KegAlign's scan wraps here too): soft-mask repeats, or use a \
             smaller -B/-C/-I (a new layout digest)",
            u32::MAX
        ));
    };
    let mut acc = 0u32;
    for s in sums.iter_mut() {
        let block = *s;
        *s = acc;
        // Cannot overflow: the total fits u32.
        acc += block;
    }
    Ok(total)
}

/// Splits the scanned hit counts into `(start_seed_index, limit_pos,
/// start_hit_val)` chunks, reproducing the `lower_bound` walk in
/// `SeedAndFilter`. Errors instead of rechunking when a walk is over-cap or
/// otherwise unsafe: new boundaries would change dedup scope.
/// `max_hits` (H) is the semantic chunk target driving the walk;
/// `hit_capacity` (C >= H) is the physical allowance for the per-chunk delta.
fn chunk_limits(
    hit_num: &[u32],
    max_hits: u32,
    hit_capacity: u32,
) -> Result<Vec<(u32, u32, u32, u32)>, String> {
    let err = |detail: &str| {
        format!(
            "unsupported chunk/cap: {detail}; rechunking is refused because it changes dedup scope"
        )
    };
    if max_hits == 0 {
        return Err(err("max_hits must be > 0"));
    }
    if hit_capacity == 0 || hit_capacity < max_hits {
        return Err(err(&format!(
            "hit_capacity {hit_capacity} must be >= max_hits {max_hits}"
        )));
    }
    if hit_num.is_empty() {
        return Ok(Vec::new());
    }
    let Ok(num_seeds) = u32::try_from(hit_num.len()) else {
        return Err(err("input length does not fit u32"));
    };
    if hit_num.windows(2).any(|w| w[1] < w[0]) {
        return Err(err("cumulative hit counts are not monotonic"));
    }

    let num_hits = hit_num[hit_num.len() - 1];
    let mut limits = Vec::new();
    let mut iter_hit_limit = max_hits;
    let mut prev: Option<u32> = None;
    for _ in 0..(u64::from(num_hits) / u64::from(max_hits)) {
        let pp = hit_num.partition_point(|&v| v < iter_hit_limit);
        if pp == 0 {
            return Err(err("lower_bound walk underflow (first bucket >= cap)"));
        }
        let Ok(pos) = u32::try_from(pp - 1) else {
            return Err(err("seed index exceeds u32"));
        };
        if prev.is_some_and(|p| pos <= p) {
            return Err(err("nonprogress seed range"));
        }
        let Some(next_limit) = hit_num[pos as usize].checked_add(max_hits) else {
            return Err(err("hit-limit add overflow"));
        };
        iter_hit_limit = next_limit;
        limits.push(pos);
        prev = Some(pos);
    }
    limits.push(num_seeds - 1);

    let mut out = Vec::with_capacity(limits.len());
    let num_chunks = limits.len();
    let mut start_seed_index = 0u32;
    let mut start_hit_val = 0u32;
    for pos in limits {
        if (pos as usize) >= hit_num.len() || pos < start_seed_index {
            return Err(err(
                "chunk seed range is empty, out of bounds, or not contiguous",
            ));
        }
        let end_hit_val = hit_num[pos as usize];
        if start_hit_val > end_hit_val {
            return Err(err("chunk hit boundaries decrease"));
        }
        if end_hit_val - start_hit_val > hit_capacity {
            return Err(err(&format!(
                "chunk hit delta exceeds capacity cap={max_hits} capacity={hit_capacity} actual={} chunk={}/{} seeds={start_seed_index}..={pos} hits={start_hit_val}..={end_hit_val} num_seeds={num_seeds} total_hits={num_hits}",
                end_hit_val - start_hit_val,
                out.len(),
                num_chunks
            )));
        }
        out.push((start_seed_index, pos, start_hit_val, end_hit_val));
        start_seed_index = pos
            .checked_add(1)
            .ok_or_else(|| err("seed index overflow"))?;
        start_hit_val = end_hit_val;
    }
    if start_seed_index != num_seeds {
        return Err(err("chunks do not cover every seed"));
    }
    Ok(out)
}

/// Shared error constructor for the chunk-walk family: `chunk_limits` keeps
/// its own inline copy (it is the untouched oracle); the sparse walk below
/// reuses this one so every message stays byte-identical.
fn chunk_err(detail: &str) -> String {
    format!("unsupported chunk/cap: {detail}; rechunking is refused because it changes dedup scope")
}

/// First scan block whose inclusive end is `>= limit`, or `None` when every
/// block ends below it (`limit > num_hits`). Binary search over the virtual
/// end array (`sums[b + 1]`, `num_hits` for the last block) — the
/// `partition_point` of the full walk, without materialising the array. The
/// ends are non-decreasing exactly when the block sums are and
/// `sums[last] <= num_hits`, both checked by the caller before the walk.
fn locate_block(sums: &[u32], num_hits: u32, limit: u32) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = sums.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let end = if mid + 1 < sums.len() {
            sums[mid + 1]
        } else {
            num_hits
        };
        if end < limit {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    (lo < sums.len()).then_some(lo)
}

/// Fetches scan block `b` through `fetch` (caching the last one, so
/// consecutive boundaries in one block fetch once) and validates what the
/// sparse data allows: exact span length, monotonicity, and containment in
/// `[sums[b], block_end]`. A last value off the block end means the device
/// and the block sums disagree, which is the same corrupt-data family as a
/// non-monotonic array, so it reports that message.
fn fetch_block(
    fetch: &mut dyn FnMut(usize) -> Result<Vec<u32>, String>,
    sums: &[u32],
    num_hits: u32,
    num_seeds: u32,
    b: usize,
    cached_block: &mut Option<usize>,
    cached_vals: &mut Vec<u32>,
) -> Result<(), String> {
    if *cached_block == Some(b) {
        return Ok(());
    }
    let vals = fetch(b)?;
    let base = b * SCAN_BLOCK as usize;
    let want = (num_seeds as usize - base).min(SCAN_BLOCK as usize);
    if vals.len() != want {
        return Err(chunk_err("fetched block length does not match seed span"));
    }
    let end = if b + 1 < sums.len() {
        sums[b + 1]
    } else {
        num_hits
    };
    if vals.windows(2).any(|w| w[1] < w[0])
        || vals.first().is_some_and(|&v| v < sums[b])
        || vals.last().is_some_and(|&v| v != end)
    {
        return Err(chunk_err("cumulative hit counts are not monotonic"));
    }
    *cached_vals = vals;
    *cached_block = Some(b);
    Ok(())
}

/// Sparse twin of [`chunk_limits`] (round 85): reproduces its walk and every
/// one of its error conditions/messages, but reads `hit_num` through two
/// levels instead of a full host copy: the boundary block is located by
/// [`locate_block`] over the host-side `sums` (the exclusive scan of the
/// per-block totals — `sums[b]` is the hit count before block `b`, covering
/// seeds `[b*SCAN_BLOCK, min((b+1)*SCAN_BLOCK, num_seeds))`), then only that
/// block's <= 256 values come through `fetch(b)` and the
/// `partition_point(|v| v < limit)` finishes inside it.
///
/// Value shortcuts that need no fetch: a boundary at a fetched block's first
/// element sits at the previous block's last seed, whose inclusive value is
/// exactly `sums[b]`; the final `limits.push(num_seeds - 1)` carries
/// `num_hits`, the total the block-sum scan produced. Both hold by
/// construction of the two-kernel scan (`scan_blocks` + `add_block_offsets`).
/// Monotonicity is checked over the block sums plus every fetched block; the
/// full-array check is impossible without the copy.
fn chunk_limits_sparse(
    fetch: &mut dyn FnMut(usize) -> Result<Vec<u32>, String>,
    sums: &[u32],
    num_seeds: u32,
    num_hits: u32,
    max_hits: u32,
    hit_capacity: u32,
) -> Result<Vec<(u32, u32, u32, u32)>, String> {
    if max_hits == 0 {
        return Err(chunk_err("max_hits must be > 0"));
    }
    if hit_capacity == 0 || hit_capacity < max_hits {
        return Err(chunk_err(&format!(
            "hit_capacity {hit_capacity} must be >= max_hits {max_hits}"
        )));
    }
    if num_seeds == 0 {
        return Ok(Vec::new());
    }
    if sums.len() != num_seeds.div_ceil(SCAN_BLOCK) as usize {
        return Err(chunk_err("block sums length does not match seed count"));
    }
    if sums.windows(2).any(|w| w[1] < w[0]) || sums.last().is_some_and(|&last| last > num_hits) {
        return Err(chunk_err("cumulative hit counts are not monotonic"));
    }

    // Last fetched block, and the hit value at each pushed boundary, so the
    // capacity pass below needs no second fetch of the same blocks.
    let mut cached_block: Option<usize> = None;
    let mut cached_vals: Vec<u32> = Vec::new();
    let mut limits: Vec<u32> = Vec::new();
    let mut boundary_vals: Vec<u32> = Vec::new();
    let mut iter_hit_limit = max_hits;
    let mut prev: Option<u32> = None;
    for _ in 0..(u64::from(num_hits) / u64::from(max_hits)) {
        let at = locate_block(sums, num_hits, iter_hit_limit);
        let pp = match at {
            // No block ends at/after the limit: every value is below it, so
            // the full walk's `partition_point` would return the length.
            None => num_seeds as usize,
            Some(b) => {
                fetch_block(
                    fetch,
                    sums,
                    num_hits,
                    num_seeds,
                    b,
                    &mut cached_block,
                    &mut cached_vals,
                )?;
                b * SCAN_BLOCK as usize + cached_vals.partition_point(|&v| v < iter_hit_limit)
            }
        };
        if pp == 0 {
            return Err(chunk_err(
                "lower_bound walk underflow (first bucket >= cap)",
            ));
        }
        let Ok(pos) = u32::try_from(pp - 1) else {
            return Err(chunk_err("seed index exceeds u32"));
        };
        if prev.is_some_and(|p| pos <= p) {
            return Err(chunk_err("nonprogress seed range"));
        }
        // Hit value at `pos` without a second fetch: the no-block arm sits at
        // `num_seeds - 1` (value `num_hits`); a boundary at a fetched block's
        // first element sits at the previous block's end (`sums[b]`, and
        // `pp > 0` means `b >= 1` there); otherwise it is the fetched
        // predecessor inside the cached block.
        let hit_val = match at {
            None => num_hits,
            Some(b) => {
                let base = b * SCAN_BLOCK as usize;
                if pp == base {
                    sums[b]
                } else {
                    cached_vals[pp - base - 1]
                }
            }
        };
        let Some(next_limit) = hit_val.checked_add(max_hits) else {
            return Err(chunk_err("hit-limit add overflow"));
        };
        iter_hit_limit = next_limit;
        limits.push(pos);
        boundary_vals.push(hit_val);
        prev = Some(pos);
    }
    limits.push(num_seeds - 1);
    boundary_vals.push(num_hits);

    let mut out = Vec::with_capacity(limits.len());
    let num_chunks = limits.len();
    let mut start_seed_index = 0u32;
    let mut start_hit_val = 0u32;
    for (i, &pos) in limits.iter().enumerate() {
        if (pos as usize) >= num_seeds as usize || pos < start_seed_index {
            return Err(chunk_err(
                "chunk seed range is empty, out of bounds, or not contiguous",
            ));
        }
        let end_hit_val = boundary_vals[i];
        if start_hit_val > end_hit_val {
            return Err(chunk_err("chunk hit boundaries decrease"));
        }
        if end_hit_val - start_hit_val > hit_capacity {
            return Err(chunk_err(&format!(
                "chunk hit delta exceeds capacity cap={max_hits} capacity={hit_capacity} actual={} chunk={}/{} seeds={start_seed_index}..={pos} hits={start_hit_val}..={end_hit_val} num_seeds={num_seeds} total_hits={num_hits}",
                end_hit_val - start_hit_val,
                out.len(),
                num_chunks
            )));
        }
        out.push((start_seed_index, pos, start_hit_val, end_hit_val));
        start_seed_index = pos
            .checked_add(1)
            .ok_or_else(|| chunk_err("seed index overflow"))?;
        start_hit_val = end_hit_val;
    }
    if start_seed_index != num_seeds {
        return Err(chunk_err("chunks do not cover every seed"));
    }
    Ok(out)
}

/// `InitializeProcessor`: `MAX_HITS_PER_GB * (totalGlobalMem / 1 GiB)`.
fn default_max_hits(ctx: &Arc<CudaContext>) -> u32 {
    let _ = ctx.bind_to_thread();
    let (_, total) = device_memory();
    let gib = if total > 0 {
        total as f32 / 1_073_741_824.0
    } else {
        1.0
    };
    ((MAX_HITS_PER_GB as f32 * gib) as u64).clamp(1, u32::MAX as u64) as u32
}

/// Hit cap and grid shared by planning, every worker, the report and a frozen
/// manifest. Resolved once; never re-derived from a worker's own device.
/// `max_hits` (H) is the semantic chunk target; `hit_capacity` (C >= H) is the
/// physical allowance that only decides success/failure, never output bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionContract {
    pub max_hits: u32,
    pub hsp_blocks: u32,
    pub hit_capacity: u32,
}

impl ExecutionContract {
    pub fn resolve(ctx: &Arc<CudaContext>, requested_max_hits: u32, hsp_blocks: u32) -> Self {
        let max_hits = resolve_max_hits(ctx, requested_max_hits);
        let hsp_blocks = if hsp_blocks > 0 {
            hsp_blocks
        } else {
            HSP_BLOCKS
        };
        Self {
            max_hits,
            hsp_blocks,
            hit_capacity: max_hits,
        }
    }

    pub fn from_resolved(max_hits: u32, hsp_blocks: u32) -> Self {
        debug_assert!(max_hits > 0);
        Self {
            max_hits,
            hsp_blocks,
            hit_capacity: max_hits,
        }
    }
}

/// Host-checked u32 grid-stride ceiling for kernel index arithmetic.
///
/// Grid-stride kernels add `stride = blockDim * gridDim` to a `u32` id, so the
/// largest physical hit count that cannot wrap is conservatively
/// `u32::MAX - max_stride`, which also covers small lane offsets (32/4) with
/// positive strides. Strides: dense score gate `hsp_blocks * NUM_WARPS * 32`,
/// materializer `hsp_blocks * NUM_WARPS`, compress `MAX_BLOCKS * MAX_THREADS`.
/// Rejects a non-positive or overflowing `hsp_blocks` grid. No kernel rewrite.
pub fn kernel_safe_hit_ceiling(hsp_blocks: u32) -> Result<u32, String> {
    if hsp_blocks == 0 {
        return Err("hsp_blocks must be positive".into());
    }
    let warps = NUM_WARPS as u64;
    let hb = u64::from(hsp_blocks);
    let dense = hb
        .checked_mul(warps)
        .and_then(|v| v.checked_mul(32))
        .ok_or_else(|| "hsp grid stride overflow".to_string())?;
    let mat = hb
        .checked_mul(warps)
        .ok_or_else(|| "hsp grid stride overflow".to_string())?;
    let compress = u64::from(MAX_BLOCKS)
        .checked_mul(u64::from(MAX_THREADS))
        .ok_or_else(|| "compress stride overflow".to_string())?;
    for s in [dense, mat, compress] {
        if s == 0 || s > u64::from(u32::MAX) {
            return Err("hsp grid stride overflow".into());
        }
    }
    let max_stride = dense.max(mat).max(compress);
    Ok(u32::MAX - max_stride as u32)
}

/// Clamps a candidate physical allowance to the kernel-safe ceiling.
/// Rejects non-positive `max_hits`, `candidate < max_hits`, and a ceiling
/// below `max_hits`. Never replans; the plan stays frozen at H.
pub fn clamp_hit_capacity(max_hits: u32, candidate: u32, hsp_blocks: u32) -> Result<u32, String> {
    if max_hits == 0 {
        return Err("max_hits must be positive".into());
    }
    if candidate < max_hits || candidate == 0 {
        return Err(format!(
            "hit_capacity {candidate} must be >= max_hits {max_hits}"
        ));
    }
    let ceiling = kernel_safe_hit_ceiling(hsp_blocks)?;
    if ceiling < max_hits {
        return Err(format!(
            "kernel-safe ceiling {ceiling} below max_hits {max_hits}"
        ));
    }
    Ok(candidate.min(ceiling))
}

/// Validates `H > 0`, `C >= H`, and kernel-safe `C` before Engine allocation.
pub fn validate_hit_config(
    max_hits: u32,
    hit_capacity: u32,
    hsp_blocks: u32,
) -> Result<(), String> {
    if max_hits == 0 {
        return Err("max_hits must be positive".into());
    }
    let clamped = clamp_hit_capacity(max_hits, hit_capacity, hsp_blocks)?;
    if clamped != hit_capacity {
        return Err(format!(
            "hit_capacity {hit_capacity} exceeds kernel-safe ceiling {clamped}"
        ));
    }
    Ok(())
}

/// A pinned request wins; `0` means "use the device-derived default".
pub fn adopt_max_hits(requested: u32, derived: u32) -> u32 {
    if requested > 0 { requested } else { derived }
}

/// Whether `rows` rows of bucket edges can be named by the k-mers' words: a
/// word is `0xFF | row << 8`, so rows 0 to 2^24 - 1, which is 2^24 of them.
/// (The test was `rows < 1 << 24` until 0.1.2 and refused the one case that
/// fits exactly, a table in which every 12-mer needs a row: a folded table of
/// a reference without rare k-mers lost its fold to it.)
#[cfg(all(
    feature = "find-hits-groups",
    feature = "ref-loc-buckets",
    feature = "gate-sieve"
))]
fn bucket_rows_fit(rows: u32) -> bool {
    rows <= 1 << 24
}

/// Round 124: `HSPZ_FOLD_COUNT=0` keeps `find_num_hits_lo`, its scan and
/// `run_bucket_counts` for folded batches; unset or `1`, `fold_seed_counts`.
#[cfg(all(
    feature = "find-hits-groups",
    feature = "ref-loc-buckets",
    feature = "gate-sieve"
))]
fn fold_count_fused() -> bool {
    static REQUEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| std::env::var("HSPZ_FOLD_COUNT").as_deref() != Ok("0"))
}

/// Round 126: survivors up to which a chunk's HSPs are taken out of the
/// survivors' records on the host (64 KiB of records).
const SMALL_TAIL: u32 = 4096;

/// Round 126: `HSPZ_TAIL=0` keeps the full `find_hsps` grid and the done scan
/// for every chunk; unset or `1`, the grid follows the survivors and a chunk
/// of at most [`SMALL_TAIL`] survivors is filtered on the host.
fn tail_lean() -> bool {
    static REQUEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| std::env::var("HSPZ_TAIL").as_deref() != Ok("0"))
}

/// Round 126: chunks whose HSPs the host filtered, and chunks that took the
/// done scan.
pub mod tail_totals {
    use std::sync::atomic::AtomicU64;
    pub static SMALL: AtomicU64 = AtomicU64::new(0);
    pub static SCANNED: AtomicU64 = AtomicU64::new(0);
}

/// Round 122: how many folded seeds a warp of the bucketed group kernels
/// holds, a power of two up to 32. A folded seed has 13 times the hits of a
/// plain one, and with 32 of them a batch is fewer warps than an A100 holds
/// resident: unit 0's writer took 2,026 ms with 32, 1,763 with 16, 1,473 with
/// 8, 1,313 with 4, 1,291 with 2 and 1,653 with 1 (the count kernel 117, 104,
/// 92, 108, 165 and 283 ms). `HSPZ_FOLD_RUN` overrides the 4.
#[cfg(all(
    feature = "find-hits-groups",
    feature = "ref-loc-buckets",
    feature = "gate-sieve"
))]
fn fold_run_len() -> u32 {
    static REQUEST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| {
        match std::env::var("HSPZ_FOLD_RUN")
            .ok()
            .map(|v| v.parse::<u32>())
        {
            None => 4,
            Some(Ok(n)) if n.is_power_of_two() && n <= 32 => n,
            Some(other) => {
                eprintln!(
                    "warning: HSPZ_FOLD_RUN={other:?} is not a power of two up to 32; using 4"
                );
                4
            }
        }
    })
}

/// `max_hits` in force: `requested`, or the device-derived default when `0`.
/// Public so the preflight can size its budget before any `Engine` exists.
pub fn resolve_max_hits(ctx: &Arc<CudaContext>, requested: u32) -> u32 {
    adopt_max_hits(requested, default_max_hits(ctx))
}

/// Tightest free VRAM across the first `n` devices. Planning must not assume
/// device 0 is the smallest card in a heterogeneous node.
pub fn min_free_bytes(n: usize) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(open_devices(n)?.1)
}

/// What [`open_devices`] returns: the contexts to hold, and the tightest free
/// VRAM among their devices.
pub type OpenDevices = (Vec<Arc<CudaContext>>, u64);

/// The primary context of each of the first `n` devices, and the tightest free
/// VRAM among them (round 112).
///
/// The contexts are returned for the caller to hold until its workers are done.
/// A primary context that is released after the probe is torn down, and the
/// worker that runs on the device builds it again: about a second per worker on
/// an A100 node, on top of 1.6 s for probing eight cards one after another.
/// While the caller holds them a worker's `CudaContext::new` only takes another
/// reference, and here they are created at the same time, one thread per
/// device.
pub fn open_devices(n: usize) -> Result<OpenDevices, DriverError> {
    let opened: Vec<Result<(Arc<CudaContext>, u64), DriverError>> = std::thread::scope(|scope| {
        let probes: Vec<_> = (0..n.max(1))
            .map(|device| {
                scope.spawn(move || {
                    // Creating the context binds it to this thread, which is
                    // what the memory query reads.
                    let ctx = CudaContext::new(device)?;
                    let (free, _) = device_memory();
                    Ok((ctx, free as u64))
                })
            })
            .collect();
        probes
            .into_iter()
            .map(|probe| probe.join().expect("device probe panicked"))
            .collect()
    });
    let mut contexts = Vec::with_capacity(opened.len());
    let mut min_free = u64::MAX;
    for device in opened {
        let (ctx, free) = device?;
        min_free = min_free.min(free);
        contexts.push(ctx);
    }
    Ok((contexts, min_free))
}

/// Visible CUDA devices.
///
/// Requires the driver to be initialised, which `CudaContext::new` does, so this is
/// only meaningful once a context exists. Returns 0 rather than an error if the
/// query fails — the caller clamps to at least one worker either way.
pub fn device_count() -> usize {
    let mut n: i32 = 0;
    // SAFETY: driver query writing one local `int`.
    unsafe { cuda_core::sys::cuDeviceGetCount(&mut n) };
    n.max(0) as usize
}

/// `(free, total)` device bytes.
pub fn device_memory() -> (usize, usize) {
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: driver query on the current context; both outputs are locals.
    unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
    (free, total)
}

/// Brackets one GPU stage: host wall time always, CUDA events when enabled.
///
/// Holds the stream by `Arc` rather than by reference so that recording into
/// `Engine::phases` does not collide with the borrow of `Engine::stream`.
struct Stage {
    stream: Arc<CudaStream>,
    start: Instant,
    events: Option<(CudaEvent, CudaEvent)>,
}

impl Stage {
    fn begin(stream: &Arc<CudaStream>, enabled: bool) -> Self {
        let events = enabled
            .then(|| {
                // CU_EVENT_DEFAULT keeps timing enabled; the default disables it.
                let timed = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
                let start = stream.context().new_event(timed).ok()?;
                let end = stream.context().new_event(timed).ok()?;
                start.record(stream).ok()?;
                Some((start, end))
            })
            .flatten();
        Stage {
            stream: stream.clone(),
            start: Instant::now(),
            events,
        }
    }

    /// Records the end event and hands the stage over *unresolved*.
    ///
    /// No host wait: measuring a stage must not force it to complete. `host` is
    /// therefore the enqueue time, not the work time — the work time is the event
    /// delta, read later. That is the whole point of the split.
    fn finish(self, name: &'static str) -> Result<PendingStage, DriverError> {
        if let Some((_, end)) = &self.events {
            end.record(&self.stream)?;
        }
        Ok(PendingStage {
            name,
            host: self.start.elapsed(),
            events: self.events,
        })
    }
}

#[cfg(test)]
mod tests {
    /// A table in which every 12-mer needs a row of bucket edges has 2^24 rows,
    /// and the last one's word still names it.
    #[cfg(all(
        feature = "find-hits-groups",
        feature = "ref-loc-buckets",
        feature = "gate-sieve"
    ))]
    #[test]
    fn every_kmer_may_need_a_row() {
        assert!(super::bucket_rows_fit(1 << 24));
        assert!(!super::bucket_rows_fit((1 << 24) + 1));
        let row = (1u32 << 24) - 1;
        let word = 0xFF | row << 8;
        assert_eq!(word >> 8, row);
        assert_eq!(word & 0xFF, 0xFF);
    }

    /// The sieve's guard: a sample every `period` launches, the plain gate
    /// after a sample that passed more than half of its hits, the sieve again
    /// after one that did not; a tiny sample decides nothing and period 0
    /// never samples.
    #[cfg(feature = "gate-sieve")]
    #[test]
    fn sieve_guard_samples_holds_and_releases() {
        use super::{GuardStep, SieveGuard};

        let run = |g: &mut SieveGuard, n: usize| -> Vec<GuardStep> {
            (0..n).map(|_| g.next_batch()).collect()
        };
        let (sample, sieve, plain) = (GuardStep::Sample, GuardStep::Sieve, GuardStep::Plain);

        let mut g = SieveGuard::new(4);
        assert_eq!(run(&mut g, 5), [sample, sieve, sieve, sieve, sample]);
        g.observe(600_000, 1_000_000);
        assert_eq!(run(&mut g, 4), [plain, plain, plain, sample]);
        // Exactly half does not hold, and a sample below the minimum is ignored.
        g.observe(500_000, 1_000_000);
        assert_eq!(run(&mut g, 4), [sieve, sieve, sieve, sample]);
        g.observe(60_000, 60_000);
        assert_eq!(run(&mut g, 3), [sieve, sieve, sieve]);

        // A hold does not outlive its period when the next sample is too small
        // to decide.
        let mut g = SieveGuard::new(4);
        assert_eq!(run(&mut g, 1), [sample]);
        g.observe(900_000, 1_000_000);
        assert_eq!(run(&mut g, 4), [plain, plain, plain, sample]);
        g.observe(1_000, 1_000);
        assert_eq!(run(&mut g, 4), [sieve, sieve, sieve, sample]);

        let mut off = SieveGuard::new(0);
        off.observe(1_000_000, 1_000_000);
        assert_eq!(run(&mut off, 3), [sieve, sieve, sieve]);
    }

    #[cfg(feature = "ref-loc-buckets")]
    use super::ref_bucket_policy;
    #[cfg(feature = "find-hits-warp")]
    use super::use_warp_find_hits;
    use super::{
        HitStats, Lifecycle, chunk_limits, chunk_limits_sparse, exclusive_scan_hit_totals,
        keep_threshold,
    };

    #[cfg(feature = "find-hits-warp")]
    #[test]
    fn warp_find_hits_cutoff_is_eight_hits_per_seed() {
        assert!(!use_warp_find_hits(7, 1));
        assert!(use_warp_find_hits(8, 1));
        assert!(!use_warp_find_hits(79, 10));
        assert!(use_warp_find_hits(80, 10));
    }

    /// `gate-int-keep`: `total >= keep_threshold(h)` is the gate's f32-rounded
    /// keep `((total as f32 as f64) as i32) >= h` for every total tried, at
    /// thresholds across the i32 range: every total within 4096 of the threshold
    /// and of each ±2^k (k = 23..31, where f32 spacing reaches and passes 1), plus
    /// 2,000,000 PRNG totals per threshold.
    #[test]
    fn integer_keep_threshold_matches_the_f32_rounded_keep() {
        const P24: i32 = 1 << 24;
        let keeps = |t: i32, h: i32| (t as f32 as f64) as i32 >= h;
        // Above 2^24 the rounding can keep a total below the threshold, or
        // reject a total equal to it.
        assert_eq!(keep_threshold(16_777_220), 16_777_219);
        assert!(keeps(16_777_219, 16_777_220) && !keeps(16_777_218, 16_777_220));
        assert_eq!(keep_threshold(P24 + 1), P24 + 2);
        assert!(!keeps(P24 + 1, P24 + 1) && keeps(P24 + 2, P24 + 1));

        let thresholds = [
            i32::MIN,
            i32::MIN + 1,
            -1,
            0,
            1,
            2000,
            3000,
            9400,
            P24 - 1,
            P24,
            P24 + 1,
            16_777_220,
            (1 << 25) - 1,
            1 << 25,
            (1 << 25) + 1,
            1 << 30,
            i32::MAX - 1,
            i32::MAX,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for h in thresholds {
            let t_min = keep_threshold(h);
            if -P24 < h && h <= P24 {
                assert_eq!(t_min, h, "every integer of this range is an f32");
            }
            let mut centres = vec![i64::from(t_min)];
            for k in 23..=31 {
                centres.extend([1i64 << k, -(1i64 << k)]);
            }
            for c in centres {
                let lo = (c - 4096).max(i64::from(i32::MIN));
                let hi = (c + 4096).min(i64::from(i32::MAX));
                for t in lo..=hi {
                    let t = t as i32;
                    assert_eq!(keeps(t, h), t >= t_min, "h {h} t {t} t_min {t_min}");
                }
            }
            for _ in 0..2_000_000 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let t = state as i32;
                assert_eq!(keeps(t, h), t >= t_min, "h {h} t {t} t_min {t_min}");
            }
        }
    }

    /// Cycle 4/5: the smallest legal window (`B <= 32`) for a 427 Mbp reference
    /// is shift 24 (16 MiB); shift 25 (32 MiB) is legal too. With the 3/4 budget
    /// the L4's 36 MiB admits 32 MiB + 3 MiB of overhead (measured 2.8% less GPU
    /// busy than 16 MiB on unit 0); the 4090 keeps 32 MiB (64 MiB + overhead
    /// exceeds 54 MiB). The T4-like and boundary cases reuse the same
    /// `ref_len`/`max_hits` and vary only `l2_bytes`, so the only thing under
    /// test is the budget arithmetic.
    #[cfg(feature = "ref-loc-buckets")]
    #[test]
    fn ref_bucket_policy_auto_offs_a_small_l2() {
        const REF_LEN: u32 = 427_000_000;
        const MAX_HITS: u32 = 16_711_680;

        // L4-like: 48 MiB L2 -> 36 MiB budget. Window 25 (32 MiB) plus the
        // ~3 MiB overhead fits; window 26 (64 MiB) does not.
        let (shift, _n, on) = ref_bucket_policy(REF_LEN, MAX_HITS, 48 << 20, false);
        assert!(on, "L4-like L2 must fit a legal window");
        assert_eq!(shift, 25, "L4-like L2 must land on the 32 MiB window");

        // A 40 MiB L2 (A100-like) -> 30 MiB budget: 32 MiB + overhead misses,
        // so it settles on 16 MiB.
        let (shift, _n, on) = ref_bucket_policy(REF_LEN, MAX_HITS, 40 << 20, false);
        assert!(on, "40 MiB L2 must fit a legal window");
        assert_eq!(shift, 24, "40 MiB L2 must land on shift 24");

        // RTX-4090-like: 72 MiB L2 -> 54 MiB budget: 32 MiB fits, 64 MiB does not.
        let (shift, _n, on) = ref_bucket_policy(REF_LEN, MAX_HITS, 72 << 20, false);
        assert!(on, "4090-like L2 must fit a legal window");
        assert_eq!(shift, 25, "4090-like L2 must land on the 32 MiB window");

        // T4-like: 4 MiB L2 cannot fit even the smallest legal (16 MiB) window.
        let (shift, _n, on) = ref_bucket_policy(REF_LEN, MAX_HITS, 4 << 20, false);
        assert!(!on, "T4-like L2 must auto-off");
        assert_eq!(shift, 24, "off still reports the smallest legal shift");

        // L2 attribute unavailable (0): off regardless of the arithmetic.
        let (_shift, _n, on) = ref_bucket_policy(REF_LEN, MAX_HITS, 0, false);
        assert!(!on, "an unavailable L2 attribute must auto-off");

        // Boundary: an L2 sized so the 16 MiB window plus overhead lands on the
        // budget exactly. `<=` must count this as a fit, not a miss.
        let overhead = u64::from(MAX_HITS) / 8 + (1 << 20);
        let exact_budget = (1u64 << 24) + overhead;
        let l2_bytes = exact_budget / 2 * 3;
        assert_eq!((l2_bytes / 3) * 2, exact_budget, "budget must hit exactly");
        let (shift, _n, on) = ref_bucket_policy(REF_LEN, MAX_HITS, l2_bytes, false);
        assert!(on, "an exact-boundary budget must count as a fit");
        assert_eq!(shift, 24, "boundary case still picks the 16 MiB window");
    }

    /// Round 106: under the sieve the window is sized for the packed planes, a
    /// quarter of a byte per base, against half of the L2, and the batch size no
    /// longer enters. An A100 (40 MiB), whose byte rule cannot fit any window at
    /// its production batch size, lands on 64 Mbp windows (16 MiB of planes, 7
    /// buckets for a 427 Mbp bin), the measured optimum; so does the L4; the
    /// 4090 takes 128 Mbp; a T4 stays off.
    #[cfg(feature = "ref-loc-buckets")]
    #[test]
    fn ref_bucket_policy_sizes_the_window_for_packed_planes() {
        const REF_LEN: u32 = 427_000_000;
        const A100_HITS: u32 = 332_401_664;

        let (_, _, on) = ref_bucket_policy(REF_LEN, A100_HITS, 40 << 20, false);
        assert!(
            !on,
            "the byte rule has no window for an A100 at production batches"
        );
        for (l2, shift, buckets) in [(40u64 << 20, 26, 7), (48 << 20, 26, 7), (72 << 20, 27, 4)] {
            let got = ref_bucket_policy(REF_LEN, A100_HITS, l2, true);
            assert_eq!(got, (shift, buckets, true), "L2 {} MiB", l2 >> 20);
        }
        // T4-like: half of 4 MiB holds no legal window even as planes.
        let (shift, _, on) = ref_bucket_policy(REF_LEN, A100_HITS, 4 << 20, true);
        assert!(!on && shift == 24);
        // The batch size does not enter the plane rule.
        assert_eq!(
            ref_bucket_policy(REF_LEN, 16_711_680, 40 << 20, true),
            ref_bucket_policy(REF_LEN, A100_HITS, 40 << 20, true)
        );
    }

    /// Round 87 (blocks): the autotune decision rule on synthetic settled
    /// block samples — median, the 3% margin, the quarter-cap settle
    /// filter — plus block closing (T_BLOCK/T_SETTLE/settled_chunks) on
    /// synthetic chunk spans. Adjacent OFF/ON blocks compare settled ns/hit:
    /// OFF ~1.64 vs ON ~1.54 => ON at ~0.94; OFF 0.349 vs ON 0.375 => OFF
    /// at ~1.07; and a 0.975 median still loses on the margin.
    #[cfg(feature = "ref-loc-buckets")]
    #[test]
    fn bucket_autotune_blocks_settle_then_decide() {
        use super::{
            AutoPhase, BUCKET_BLOCKS, BUCKET_WARM_CHUNKS, BlockSample, BucketMode,
            block_pair_ratios, bucket_trial_eligible, decide_bucket_autotune, median_ns_per_hit,
        };

        assert_eq!(BUCKET_BLOCKS, 6, "OFF, ON, OFF, ON, OFF, ON");
        assert_eq!(
            BUCKET_WARM_CHUNKS, 4,
            "first 4 eligible chunks run OFF untimed"
        );
        assert_eq!(median_ns_per_hit(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_ns_per_hit(&[4.0, 1.0, 2.0, 3.0]), 2.5);

        let samples = |off: &[f64; 3], on: &[f64; 3]| -> Vec<BlockSample> {
            let mut v = Vec::new();
            for k in 0..3 {
                for (ns, is_on) in [(off[k], false), (on[k], true)] {
                    v.push(BlockSample {
                        on: is_on,
                        ns_per_hit: ns,
                        chunks: 6,
                        settled_chunks: 4,
                        hits: 4_000_000,
                        span_ms: 60.0,
                    });
                }
            }
            v
        };
        // OFF ~1.64 vs ON ~1.54 => ON at ~0.94.
        let ratios = block_pair_ratios(&samples(&[1.64, 1.63, 1.65], &[1.53, 1.55, 1.54]));
        assert_eq!(ratios.len(), 3);
        assert!(
            (median_ns_per_hit(&ratios) - 0.94).abs() < 0.01,
            "ratios {ratios:?}"
        );
        assert!(decide_bucket_autotune(&ratios));
        // OFF 0.349 vs ON 0.375 => OFF at ~1.07.
        let ratios = block_pair_ratios(&samples(&[0.349, 0.349, 0.349], &[0.375, 0.375, 0.375]));
        assert!(
            (median_ns_per_hit(&ratios) - 1.07).abs() < 0.01,
            "ratios {ratios:?}"
        );
        assert!(!decide_bucket_autotune(&ratios));
        // A 0.975 median still loses on the 3% margin => OFF.
        assert!(!decide_bucket_autotune(&[0.975; 3]));
        assert!(!decide_bucket_autotune(&[0.97; 3]));
        assert!(!decide_bucket_autotune(&[1.0; 3]));
        assert!(decide_bucket_autotune(&[0.969; 3]));

        const MAX: u32 = 16_711_680;
        assert!(!bucket_trial_eligible(MAX / 4 - 1, MAX));
        assert!(bucket_trial_eligible(MAX / 4, MAX));
        assert!(!bucket_trial_eligible(0, MAX));

        // Block mechanics on synthetic chunk spans (T_BLOCK=60, T_SETTLE=15).
        const T_BLOCK: f64 = 60.0;
        const T_SETTLE: f64 = 15.0;
        let mut mode = BucketMode::Auto {
            phase: AutoPhase::Warm { n: 0 },
            blocks: Vec::new(),
        };
        // Warm: the first 4 eligible chunks run OFF untimed; an ineligible
        // chunk neither times nor advances the count.
        assert!(!mode.auto_chunk(1, MAX));
        assert!(!mode.production_path());
        for _ in 0..3 {
            assert!(!mode.auto_chunk(MAX, MAX));
            assert!(!mode.production_path());
        }
        assert!(!mode.auto_chunk(MAX, MAX));
        // Block 0 runs OFF timed; an engine that ends here stays in Auto.
        assert!(mode.auto_chunk(MAX, MAX));
        assert!(!mode.production_path());
        assert!(matches!(mode, BucketMode::Auto { .. }));
        // Two pre-settle chunks join the span but not the settled totals;
        // a tail joins the span but is never settled.
        assert_eq!(
            mode.commit_block_span(10.0, MAX, MAX, T_BLOCK, T_SETTLE),
            (None, None, None)
        );
        assert_eq!(
            mode.commit_block_span(10.0, MAX, MAX, T_BLOCK, T_SETTLE),
            (None, None, None)
        );
        assert_eq!(
            mode.commit_block_span(10.0, 1, MAX, T_BLOCK, T_SETTLE),
            (None, None, None)
        );
        assert_eq!(
            mode.commit_block_span(10.0, MAX, MAX, T_BLOCK, T_SETTLE),
            (None, None, None)
        );
        assert_eq!(
            mode.commit_block_span(10.0, MAX, MAX, T_BLOCK, T_SETTLE),
            (None, None, None)
        );
        assert_eq!(
            mode.commit_block_span(10.0, MAX, MAX, T_BLOCK, T_SETTLE),
            (None, None, None)
        );
        // Span 60 ms but only 3 settled: both conditions are required, so
        // the next chunk closes block 0.
        let (closed, decided, inconclusive) =
            mode.commit_block_span(10.0, MAX, MAX, T_BLOCK, T_SETTLE);
        assert_eq!(decided, None);
        assert_eq!(inconclusive, None);
        let Some((k, sample)) = closed else {
            panic!("70 ms span with 4 settled chunks must close block 0");
        };
        assert_eq!(k, 0);
        assert!(!sample.on);
        assert_eq!((sample.chunks, sample.settled_chunks), (7, 4));
        assert_eq!(sample.hits, 4 * u64::from(MAX));
        assert_eq!(sample.span_ms, 70.0);
        assert_eq!(sample.ns_per_hit, 40.0 * 1e6 / (4 * MAX) as f64);
        // Next block alternates to ON; the engine is still undecided.
        assert!(matches!(mode, BucketMode::Auto { .. }));
        assert!(mode.production_path());
        assert!(mode.auto_chunk(MAX, MAX));

        // Run the remaining 5 blocks (OFF 10 ms/chunk, ON 9 ms/chunk) to a
        // decision: ON settles ~10% cheaper, so the verdict must be ON.
        let mut last_decision = None;
        for expect_k in 1..BUCKET_BLOCKS {
            let expect_on = expect_k % 2 == 1;
            let span = if expect_on { 9.0 } else { 10.0 };
            let (closed, decided, _) = loop {
                let (closed, decided, inconclusive) =
                    mode.commit_block_span(span, MAX, MAX, T_BLOCK, T_SETTLE);
                assert!(
                    inconclusive.is_none(),
                    "settled path must not go inconclusive"
                );
                if closed.is_some() || decided.is_some() {
                    break (closed, decided, inconclusive);
                }
            };
            let Some((k, sample)) = closed else {
                panic!("block {expect_k} never closed");
            };
            assert_eq!(k, expect_k);
            assert_eq!(sample.on, expect_on);
            last_decision = decided;
            if expect_k < BUCKET_BLOCKS - 1 {
                assert!(last_decision.is_none());
                assert!(matches!(mode, BucketMode::Auto { .. }));
            }
        }
        let Some(d) = last_decision else {
            panic!("6 blocks must decide");
        };
        assert!(d.use_on, "ratios {:?}", d.ratios);
        assert_eq!(d.ratios.len(), 3);
        assert_eq!(d.off_ns.len(), 3);
        assert_eq!(d.on_ns.len(), 3);
        // Decided chunks run the pinned path and never time.
        assert!(mode.production_path());
        assert!(!mode.auto_chunk(MAX, MAX));
        // Forced/Decided never time.
        assert!(!BucketMode::Forced(true).auto_chunk(MAX, MAX));
        assert!(BucketMode::Forced(true).production_path());
        assert!(!BucketMode::Decided(false).auto_chunk(MAX, MAX));
        assert!(!BucketMode::Decided(false).production_path());
    }

    /// Inconclusive OFF blocks (span >= 2*T_BLOCK, <4 settled) are dropped
    /// and a fresh OFF block starts at the same index; the trial stays in
    /// Auto with no decided path.
    #[cfg(feature = "ref-loc-buckets")]
    #[test]
    fn bucket_inconclusive_off_drops_and_retries_same_block() {
        use super::{AutoPhase, BucketMode, InconclusiveBlock};
        const T_BLOCK: f64 = 60.0;
        const T_SETTLE: f64 = 15.0;
        const MAX: u32 = 16_711_680;
        let mut mode = BucketMode::Auto {
            phase: AutoPhase::Block {
                on: false,
                k: 0,
                span_ms: 0.0,
                settled_span_ms: 0.0,
                settled_hits: 0,
                chunks: 0,
                settled_chunks: 0,
            },
            blocks: Vec::new(),
        };
        // 11 tail chunks (ineligible): span 110 < 120, no decision, no drop.
        for _ in 0..11 {
            assert_eq!(
                mode.commit_block_span(10.0, 1, MAX, T_BLOCK, T_SETTLE),
                (None, None, None)
            );
        }
        // 12th tail chunk: span 120 >= 2*T_BLOCK with 0 settled -> drop.
        let (closed, decided, inconclusive) =
            mode.commit_block_span(10.0, 1, MAX, T_BLOCK, T_SETTLE);
        assert_eq!(closed, None);
        assert_eq!(decided, None);
        assert_eq!(
            inconclusive,
            Some(InconclusiveBlock {
                k: 0,
                on: false,
                span_ms: 120.0,
                settled_chunks: 0,
            })
        );
        // Dropped: no block recorded, fresh OFF block at the same index.
        assert!(matches!(mode, BucketMode::Auto { .. }));
        assert_eq!(mode.undecided_blocks(), Some(0));
        assert!(!mode.production_path());
        assert!(mode.auto_chunk(MAX, MAX));
        // The retry can still close conclusively.
        for _ in 0..5 {
            let (_, _, inc) = mode.commit_block_span(10.0, MAX, MAX, T_BLOCK, T_SETTLE);
            assert_eq!(inc, None);
        }
        let (closed, decided, inc) = mode.commit_block_span(10.0, MAX, MAX, T_BLOCK, T_SETTLE);
        assert_eq!(inc, None);
        assert_eq!(decided, None);
        let Some((k, sample)) = closed else {
            panic!("retried OFF block must close");
        };
        assert_eq!(k, 0);
        assert!(!sample.on);
        assert_eq!(mode.undecided_blocks(), Some(1));
    }

    /// An inconclusive ON block ends the trial as `Decided(false)`: the
    /// production default path, with the ledger fields for the
    /// `auto -> off (inconclusive: ...)` line.
    #[cfg(feature = "ref-loc-buckets")]
    #[test]
    fn bucket_inconclusive_on_falls_back_to_off() {
        use super::{AutoPhase, BlockSample, BucketMode, InconclusiveBlock};
        const T_BLOCK: f64 = 60.0;
        const T_SETTLE: f64 = 15.0;
        const MAX: u32 = 16_711_680;
        let mut mode = BucketMode::Auto {
            phase: AutoPhase::Block {
                on: true,
                k: 1,
                span_ms: 0.0,
                settled_span_ms: 0.0,
                settled_hits: 0,
                chunks: 0,
                settled_chunks: 0,
            },
            blocks: vec![BlockSample {
                on: false,
                ns_per_hit: 1.0,
                chunks: 7,
                settled_chunks: 4,
                hits: 4 * u64::from(MAX),
                span_ms: 70.0,
            }],
        };
        for _ in 0..11 {
            assert_eq!(
                mode.commit_block_span(10.0, 1, MAX, T_BLOCK, T_SETTLE),
                (None, None, None)
            );
        }
        let (closed, decided, inconclusive) =
            mode.commit_block_span(10.0, 1, MAX, T_BLOCK, T_SETTLE);
        assert_eq!(closed, None);
        assert_eq!(decided, None);
        assert_eq!(
            inconclusive,
            Some(InconclusiveBlock {
                k: 1,
                on: true,
                span_ms: 120.0,
                settled_chunks: 0,
            })
        );
        assert_eq!(mode, BucketMode::Decided(false));
        assert!(!mode.production_path());
        assert!(!mode.auto_chunk(MAX, MAX));
        assert_eq!(mode.undecided_blocks(), None);
    }

    /// An engine that finishes in `Auto` reports its closed-block count so
    /// the ledger can say why there is no decision; forced/decided engines
    /// report nothing.
    #[cfg(feature = "ref-loc-buckets")]
    #[test]
    fn bucket_engine_end_reports_undecided_blocks() {
        use super::{AutoPhase, BucketMode};
        let warm = BucketMode::Auto {
            phase: AutoPhase::Warm { n: 0 },
            blocks: Vec::new(),
        };
        assert_eq!(warm.undecided_blocks(), Some(0));
        assert_eq!(BucketMode::Forced(true).undecided_blocks(), None);
        assert_eq!(BucketMode::Forced(false).undecided_blocks(), None);
        assert_eq!(BucketMode::Decided(true).undecided_blocks(), None);
        assert_eq!(BucketMode::Decided(false).undecided_blocks(), None);
    }

    /// Two distributions with the *same* mean hits/seed, the same seed count
    /// and the same total work must come out with very different numbers, or
    /// these statistics cannot do the job the mean was failing at. This checks
    /// separation only — lane utilisation is descriptive, not a predictor of
    /// which mapping is faster.
    #[test]
    fn density_statistics_separate_uniform_density_from_a_repeat_tail() {
        // Uniformly dense: 64 seeds of 32 hits each. Every warp step is full
        // under both mappings, so both should read ~100%.
        let mut uniform = HitStats::default();
        for _ in 0..64 {
            uniform.observe(32);
        }
        let (t_uni, w_uni) = uniform.lane_utilisation();
        assert!(
            t_uni > 99.0,
            "thread-per-seed wastes nothing here, got {t_uni}"
        );
        assert!(
            w_uni > 99.0,
            "warp-per-seed wastes nothing here, got {w_uni}"
        );

        // Same 2048 hits and same 64 seeds, but carried by one repeat bucket:
        // 63 seeds of 1 hit and one seed of 1985.
        let mut tailed = HitStats::default();
        for _ in 0..63 {
            tailed.observe(1);
        }
        tailed.observe(1985);
        assert_eq!(tailed.total_hits, uniform.total_hits, "same total work");
        assert_eq!(tailed.seeds(), uniform.seeds(), "same seed count");

        let (t_tail, w_tail) = tailed.lane_utilisation();
        // Thread-per-seed: both groups of 32 run until the longest walk ends,
        // so the 1985-hit seed drags 31 idle lanes behind it.
        assert!(
            t_tail < 5.0,
            "thread-per-seed should collapse here, got {t_tail}"
        );
        // Warp-per-seed: the 63 one-hit seeds each burn a whole warp step, but
        // the repeat seed runs at full width, so it recovers most of the loss.
        assert!(
            w_tail > t_tail * 4.0,
            "warp mapping should win at a tail: {w_tail} vs {t_tail}"
        );
        assert!(
            w_tail < 99.0,
            "the one-hit seeds still waste lanes, got {w_tail}"
        );

        // And the hit-weighted share is what exposes the tail: nearly all the
        // work sits in warp-filling seeds even though 63 of 64 seeds do not.
        assert!(tailed.hit_share_warp_filling() > 95.0);
        assert_eq!(uniform.hit_share_warp_filling(), 0.0, "r=32 is not >=33");
    }

    /// A trailing partial group must still be charged, or `thread_slots`
    /// silently drops the last <32 seeds.
    #[test]
    fn partial_warp_group_is_charged() {
        let mut h = HitStats::default();
        h.observe(10);
        let (t, w) = h.lane_utilisation();
        assert_eq!(w, 10.0 / 32.0 * 100.0, "one warp step for one seed");
        assert_eq!(t, 10.0 / 320.0 * 100.0, "32 lanes stepping 10 times");
        // Repeated calls must not double-charge.
        assert_eq!(h.lane_utilisation(), (t, w));
    }

    #[test]
    fn thread_lane_slots_restart_at_launch_boundaries() {
        let mut h = HitStats::default();
        h.observe_launch(&[1]);
        h.observe_launch(&[2]);
        let (thread, warp) = h.lane_utilisation();
        assert_eq!(thread, 3.0 / 96.0 * 100.0);
        assert_eq!(warp, 3.0 / 64.0 * 100.0);
    }
    #[cfg(feature = "dense-anchors")]
    use super::SCAN_BLOCK;

    /// The counters exist to catch a reference index rebuilt
    /// per work unit instead of per reference bin — a regression that changes no
    /// output and shows up only as ~1 GB of extra H->D traffic per work unit.
    #[test]
    fn lifecycle_rejects_per_work_unit_reference_rebuilds() {
        // A 2x3 plan: 2 reference bins, 6 work units.
        let good = Lifecycle {
            seed_table_builds: 2,
            index_loads: 0,
            engine_creations: 2,
            reference_uploads: 2,
            query_swaps: 6,
            work_units_executed: 6,
        };
        assert!(good.check(2, 6).is_ok(), "{:?}", good.check(2, 6));

        // An indexed run loads instead of building: 2 loads, 0 builds.
        let indexed = Lifecycle {
            seed_table_builds: 0,
            index_loads: 2,
            ..good
        };
        assert!(indexed.check(2, 6).is_ok(), "{:?}", indexed.check(2, 6));

        // Mixed builds and loads still sum to the visit count; anything else
        // fails, and a visit is never both.
        let mixed = Lifecycle {
            seed_table_builds: 1,
            index_loads: 1,
            ..good
        };
        assert!(mixed.check(2, 6).is_ok(), "{:?}", mixed.check(2, 6));
        let both = Lifecycle {
            seed_table_builds: 2,
            index_loads: 2,
            ..good
        };
        assert!(both.check(2, 6).is_err(), "a visit is never both");

        // The exact regression this guards against: an Engine per work unit.
        let per_unit = Lifecycle {
            engine_creations: 6,
            reference_uploads: 6,
            ..good
        };
        let err = per_unit
            .check(2, 6)
            .expect_err("must reject 6 uploads for 2 bins");
        assert!(err.contains("engine_creations"), "{err}");

        // A query bin that never swapped: the silent-zero-HSP shape.
        let missed_swap = Lifecycle {
            query_swaps: 5,
            ..good
        };
        assert!(
            missed_swap.check(2, 6).is_err(),
            "a skipped swap must not pass"
        );
    }

    #[test]
    fn pinned_hit_cap_is_not_replaced_by_device_derivation() {
        assert_eq!(super::adopt_max_hits(99_165_440, 16_711_680), 99_165_440);
        assert_eq!(super::adopt_max_hits(0, 16_711_680), 16_711_680);
        let c = super::ExecutionContract::from_resolved(99_165_440, 16384);
        assert_eq!(c.max_hits, 99_165_440);
        assert_eq!(c.hit_capacity, 99_165_440, "capacity defaults to H");
    }

    /// A batch whose hits exceed `u32::MAX` must be refused, never wrapped: the
    /// wrapped value of this one (16,399,513) sits below a 16.7 M cap and would
    /// take the unchecked single-chunk path. Batches that fit scan as before.
    #[test]
    fn batch_hit_total_beyond_u32_is_an_error_not_a_wrap() {
        let mut sums = vec![3u32, 0, 5, 7];
        assert_eq!(exclusive_scan_hit_totals(&mut sums), Ok(15));
        assert_eq!(sums, vec![0, 3, 3, 8]);

        let mut sums = vec![u32::MAX - 1, 1];
        assert_eq!(exclusive_scan_hit_totals(&mut sums), Ok(u32::MAX));
        assert_eq!(sums, vec![0, u32::MAX - 1]);

        let mut sums = vec![u32::MAX, 16_399_514];
        let err = exclusive_scan_hit_totals(&mut sums).unwrap_err();
        assert!(err.contains("4311366809"), "{err}");
        assert!(err.contains("u32 hit index"), "{err}");
    }

    #[test]
    fn single_chunk_when_hits_fit() {
        let hit_num = vec![2, 5, 9];
        assert_eq!(
            chunk_limits(&hit_num, 1000, 1000).unwrap(),
            vec![(0, 2, 0, 9)]
        );
    }

    #[test]
    fn chunk_boundaries_follow_the_lower_bound_walk() {
        // 4 seeds, running totals 3/7/11/15, MAX_HITS = 5.
        let hit_num = vec![3, 7, 11, 15];
        let chunks = chunk_limits(&hit_num, 5, 5).unwrap();
        assert_eq!(
            chunks,
            vec![(0, 0, 0, 3), (1, 1, 3, 7), (2, 2, 7, 11), (3, 3, 11, 15)]
        );
        for w in chunks.windows(2) {
            assert_eq!(w[1].0, w[0].1 + 1, "chunks tile the seed range");
        }
        assert_eq!(
            chunks.last().unwrap().1,
            3,
            "last chunk ends at num_seeds-1"
        );
    }

    #[test]
    fn chunk_limits_sparse_matches_full_walk_on_random_batches() {
        use super::SCAN_BLOCK;
        // Deterministic xorshift64: no RNG crate needed for a cfg(test) check.
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut next = || -> u64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let scan = SCAN_BLOCK as usize;
        let mut done = 0usize;
        let mut guard = 0u32;
        let mut total_fetches = 0u64;
        let mut total_boundaries = 0u64;
        let mut ok_fetches = 0u64;
        let mut ok_boundaries = 0u64;
        let mut max_per_boundary = 0f64;
        while done < 2000 {
            guard += 1;
            assert!(guard < 1_000_000, "random batch generator is stuck");
            let n = 1 + (next() % 3000) as usize;
            let style = next() % 8;
            // Counts drawn so many seeds have 0 hits (runs of EQUAL cumulative
            // values); one batch in four also gets a huge single seed.
            let huge_idx = (next() % n as u64) as usize;
            let huge = next() % 4 == 0;
            let huge_val = 1 + next() % 40_000_000;
            let mut acc = 0u64;
            let mut full: Vec<u32> = Vec::with_capacity(n);
            for i in 0..n {
                let mut c = match style {
                    0 => {
                        if next() % 4 == 0 {
                            next() % 20
                        } else {
                            0
                        }
                    }
                    1 => {
                        if next() % 3 == 0 {
                            next() % 3
                        } else {
                            0
                        }
                    }
                    2 => next() % 100,
                    3 => {
                        if next() % 10 == 0 {
                            500 + next() % 5000
                        } else {
                            next() % 5
                        }
                    }
                    _ => {
                        if next() % 2 == 0 {
                            0
                        } else {
                            1 + next() % 50
                        }
                    }
                };
                if huge && i == huge_idx {
                    c = huge_val;
                }
                acc += c;
                if acc > u64::from(u32::MAX) {
                    break;
                }
                full.push(acc as u32);
            }
            if full.len() != n {
                continue; // u32 overflow draw: regenerate.
            }
            let total = acc;
            // Block sums (exclusive scan of the per-block totals), exactly as
            // the host builds them from the device block sums.
            let blocks = n.div_ceil(scan);
            let mut sums = vec![0u32; blocks];
            let mut run = 0u32;
            for b in 0..blocks {
                sums[b] = run;
                let hi = ((b + 1) * scan).min(n);
                let first = if b == 0 { 0 } else { full[b * scan - 1] };
                run += full[hi - 1] - first;
            }
            // max_hits: tiny (many chunks), total-based, exactly on a value,
            // exactly on a block edge, exactly the total, huge-based, or 0.
            let max_hits = match next() % 8 {
                0 => (1 + next() % 50) as u32,
                1 => (total / (1 + next() % 5)).clamp(1, u64::from(u32::MAX)) as u32,
                2 => full[(next() % n as u64) as usize],
                3 => {
                    let edges = [255, 256, 511, 512, 1023, 1024];
                    edges
                        .iter()
                        .find(|&&e| e < n)
                        .map(|&e| full[e])
                        .unwrap_or_else(|| full[n - 1])
                }
                4 => total.clamp(1, u64::from(u32::MAX)) as u32,
                5 => (huge_val / 2).clamp(1, u64::from(u32::MAX)) as u32,
                6 => 0,
                _ => 1,
            };
            let hit_capacity = match next() % 6 {
                0 => max_hits,
                1 => max_hits.saturating_add((next() % 1000) as u32),
                2 => max_hits.saturating_add(50_000_000),
                3 => max_hits.saturating_sub(1),
                4 => 0,
                _ => max_hits,
            };
            let mut fetches = 0usize;
            let mut fetch = |b: usize| -> Result<Vec<u32>, String> {
                fetches += 1;
                let lo = b * scan;
                let hi = ((b + 1) * scan).min(n);
                assert!(lo < hi, "sparse walk fetched out-of-range block {b}");
                Ok(full[lo..hi].to_vec())
            };
            let expected = chunk_limits(&full, max_hits, hit_capacity);
            let got = chunk_limits_sparse(
                &mut fetch,
                &sums,
                n as u32,
                total as u32,
                max_hits,
                hit_capacity,
            );
            assert_eq!(
                got, expected,
                "mismatch case {done} n={n} H={max_hits} C={hit_capacity} style={style}"
            );
            // Walk iterations plus the final boundary bound the block traffic:
            // each iteration fetches at most one new block.
            let boundaries = match &expected {
                Ok(v) => v.len().max(1),
                Err(_) if max_hits == 0 => 1,
                Err(_) => (total / u64::from(max_hits)) as usize + 1,
            };
            assert!(
                fetches <= 2 * boundaries,
                "case {done}: {fetches} block fetches for {boundaries} boundaries"
            );
            total_fetches += fetches as u64;
            total_boundaries += boundaries as u64;
            if expected.is_ok() {
                ok_fetches += fetches as u64;
                ok_boundaries += boundaries as u64;
            }
            max_per_boundary = max_per_boundary.max(fetches as f64 / boundaries as f64);
            done += 1;
        }
        eprintln!(
            "sparse-walk property test: {done} cases, {total_fetches} block fetches \
             for {total_boundaries} walk-bound boundaries (avg {:.3}/boundary, max {:.3}/boundary); \
             ok-cases: {ok_fetches} fetches for {ok_boundaries} exact boundaries (avg {:.3}/boundary)",
            total_fetches as f64 / total_boundaries as f64,
            max_per_boundary,
            ok_fetches as f64 / ok_boundaries as f64,
        );
    }

    #[test]
    fn chunk_limits_sparse_matches_full_walk_on_block_edge_totals() {
        use super::SCAN_BLOCK;
        const H: u32 = 1000;
        const C: u32 = 3000;
        // Block-end totals: interior seeds step by +1 while each block's first
        // seed jumps by 999. A block j can therefore end exactly on the value
        // the walk seeks, so the next limit sums[b] + H is already crossed by
        // block b's FIRST seed: the sparse walk must resolve the boundary at
        // pos = b*SCAN_BLOCK - 1 with hit value sums[b] from local pp == 0,
        // an alignment the random test does not force to actually occur.
        for nb in [2usize, 3, 5] {
            for partial_last in [false, true] {
                let span = |b: usize| -> usize {
                    if b + 1 == nb && partial_last {
                        17
                    } else {
                        SCAN_BLOCK as usize
                    }
                };
                let n: usize = (0..nb).map(span).sum();
                let mut full = Vec::with_capacity(n);
                let mut acc = 0u32;
                for b in 0..nb {
                    for i in 0..span(b) {
                        acc += if b > 0 && i == 0 { 999 } else { 1 };
                        full.push(acc);
                    }
                }
                // Exclusive scan of the per-block totals, exactly as the host
                // builds `sums` for the sparse walk.
                let mut sums = Vec::with_capacity(nb);
                let mut run = 0u32;
                let mut seeds = 0usize;
                for b in 0..nb {
                    sums.push(run);
                    let next = seeds + span(b);
                    run += full[next - 1] - if seeds == 0 { 0 } else { full[seeds - 1] };
                    seeds = next;
                }
                let mut fetches = 0usize;
                let mut fetch = |b: usize| -> Result<Vec<u32>, String> {
                    fetches += 1;
                    let lo = b * SCAN_BLOCK as usize;
                    Ok(full[lo..(lo + SCAN_BLOCK as usize).min(full.len())].to_vec())
                };
                let expected = chunk_limits(&full, H, C).unwrap();
                let num_seeds = seeds as u32;
                let got = chunk_limits_sparse(&mut fetch, &sums, num_seeds, acc, H, C).unwrap();
                assert_eq!(got, expected, "nb={nb} partial_last={partial_last}");
                let end = got.last().unwrap().1;
                assert!(
                    got.iter()
                        .any(|&(_, pos, _, _)| pos % SCAN_BLOCK == SCAN_BLOCK - 1 && pos != end),
                    "nb={nb} partial_last={partial_last}: no local-pp-0 boundary"
                );
                assert!(
                    fetches <= got.len(),
                    "nb={nb} partial_last={partial_last}: {fetches} fetches"
                );
            }
        }
    }

    #[test]
    fn chunk_limits_sparse_no_block_arm_matches_full_partition_point() {
        use super::{SCAN_BLOCK, locate_block};
        // 600 seeds: a flat run of 5s crossing a block edge, then +7/seed.
        let n = 600usize;
        let full: Vec<u32> = (0..n)
            .map(|i| if i < 300 { 5 } else { 5 + (i as u32 - 299) * 7 })
            .collect();
        let scan = SCAN_BLOCK as usize;
        let blocks = n.div_ceil(scan);
        let mut sums = vec![0u32; blocks];
        let mut run = 0u32;
        for b in 0..blocks {
            sums[b] = run;
            let hi = ((b + 1) * scan).min(n);
            let first = if b == 0 { 0 } else { full[b * scan - 1] };
            run += full[hi - 1] - first;
        }
        let num_hits = full[n - 1];
        // A limit past the total: the block-end search says "no block", and
        // the full walk's partition_point says "the length" — same position.
        let limit = num_hits + 1;
        assert_eq!(locate_block(&sums, num_hits, limit), None);
        assert_eq!(
            full.partition_point(|&v| v < limit),
            n,
            "full walk pp for limit > num_hits is the length"
        );
        // And the sparse/full walks agree on the array itself.
        let mut fetches = 0usize;
        let mut fetch = |b: usize| -> Result<Vec<u32>, String> {
            fetches += 1;
            Ok(full[b * scan..((b + 1) * scan).min(n)].to_vec())
        };
        assert_eq!(
            chunk_limits_sparse(&mut fetch, &sums, n as u32, num_hits, 500, 600),
            chunk_limits(&full, 500, 600),
        );
        let _ = fetches;
    }

    #[test]
    fn chunk_limits_rejects_unsupported_chunk_cap() {
        let reject = |hits: &[u32], cap: u32, capacity: u32| {
            let err = chunk_limits(hits, cap, capacity).expect_err("must reject");
            assert!(err.contains("unsupported chunk/cap"), "{err}");
            assert!(
                err.contains("rechunking is refused because it changes dedup scope"),
                "{err}"
            );
        };
        reject(&[4, 8, 12, 16], 6, 6);
        reject(&[8, 12, 16], 6, 6);
        reject(&[3, 15], 6, 6);
        reject(&[1], 0, 0);
        reject(&[], 0, 0);
        reject(&[5, 3, 8], 10, 10);
        // C < H or zero capacity fails even when the H walk would fit.
        reject(&[2, 5, 9], 1000, 999);
        reject(&[2, 5, 9], 1000, 0);
        assert!(chunk_limits(&[], 6, 6).unwrap().is_empty());
    }

    #[test]
    fn chunk_limits_overcap_reports_actionable_context() {
        let err = chunk_limits(&[4, 8, 12, 16], 6, 6).expect_err("must reject");
        assert!(err.contains("chunk hit delta exceeds capacity"), "{err}");
        assert!(err.contains("cap=6"), "{err}");
        assert!(err.contains("capacity=6"), "{err}");
        assert!(err.contains("actual=8"), "{err}");
        assert!(err.contains("2..=3"), "{err}");
        assert!(err.contains("8..=16"), "{err}");
        assert!(err.contains("num_seeds=4"), "{err}");
        assert!(err.contains("total_hits=16"), "{err}");
        // Same walk at H=6, C=8 preserves the 3 historical chunks.
        assert_eq!(
            chunk_limits(&[4, 8, 12, 16], 6, 8).unwrap(),
            vec![(0, 0, 0, 4), (1, 1, 4, 8), (2, 3, 8, 16)]
        );
        assert_eq!(
            chunk_limits(&[2, 5, 9], 1000, 1000).unwrap(),
            vec![(0, 2, 0, 9)]
        );
        assert_eq!(
            chunk_limits(&[3, 7, 11, 15], 5, 5).unwrap(),
            vec![(0, 0, 0, 3), (1, 1, 3, 7), (2, 2, 7, 11), (3, 3, 11, 15)]
        );
    }

    #[test]
    fn chunk_limits_near_u32_limit_does_not_panic() {
        assert!(chunk_limits(&[3, u32::MAX], u32::MAX - 1, u32::MAX - 1).is_err());
        assert!(chunk_limits(&[u32::MAX], u32::MAX, u32::MAX).is_err());
        assert_eq!(
            chunk_limits(&[0, u32::MAX], u32::MAX, u32::MAX).unwrap(),
            vec![(0, 0, 0, 0), (1, 1, 0, u32::MAX)]
        );
    }

    #[test]
    fn tail_capacity_exact_delta_hit_and_seed_boundaries() {
        // Tail: H=16711680, total=33423347, first chunk hits 16711642 (H-38),
        // final delta 16711705 (H+25). Synthetic cumulative Vec with N=1198054.
        const H: u32 = 16_711_680;
        const N: usize = 1_198_054;
        const FIRST: u32 = 16_711_642;
        const TOTAL: u32 = 33_423_347;
        let mut cumulative = Vec::with_capacity(N);
        for i in 0..N {
            if i <= 623_634 {
                cumulative.push(FIRST);
            } else if i < N - 1 {
                cumulative.push(H);
            } else {
                cumulative.push(TOTAL);
            }
        }
        assert_eq!(cumulative.len(), N);
        // C=H must fail on the H+25 final delta.
        let err = chunk_limits(&cumulative, H, H).expect_err("C=H must fail");
        assert!(err.contains("capacity=16711680"), "{err}");
        assert!(err.contains("actual=16711705"), "{err}");
        // C=H+25 admits exactly the historical two chunks.
        let chunks = chunk_limits(&cumulative, H, H + 25).unwrap();
        assert_eq!(chunk_limits(&cumulative, H, TOTAL + 1).unwrap(), chunks);
        assert_eq!(
            chunks,
            vec![
                (0, 623634, 0, 16_711_642),
                (623635, 1198053, 16_711_642, 33_423_347)
            ]
        );
    }

    #[test]
    fn kernel_stride_ceiling_and_capacity_validation() {
        use super::{HSP_BLOCKS, MAX_BLOCKS, MAX_THREADS, NUM_WARPS};
        use super::{clamp_hit_capacity, kernel_safe_hit_ceiling, validate_hit_config};
        // Legacy constructors preserve H and default capacity to H.
        let c = super::ExecutionContract::from_resolved(16_711_680, HSP_BLOCKS);
        assert_eq!((c.max_hits, c.hit_capacity), (16_711_680, 16_711_680));
        // Safe ceiling is u32::MAX minus the largest grid stride.
        let ceiling = kernel_safe_hit_ceiling(HSP_BLOCKS).unwrap();
        let dense = HSP_BLOCKS as u64 * NUM_WARPS as u64 * 32;
        let mat = HSP_BLOCKS as u64 * NUM_WARPS as u64;
        let compress = MAX_BLOCKS as u64 * MAX_THREADS as u64;
        assert_eq!(ceiling, u32::MAX - dense.max(mat).max(compress) as u32);
        assert!(ceiling >= 16_711_680, "ceiling must admit H");
        // Invalid grids and bad capacities fail without a GPU.
        assert!(kernel_safe_hit_ceiling(0).is_err());
        assert!(kernel_safe_hit_ceiling(u32::MAX).is_err());
        assert!(clamp_hit_capacity(ceiling + 1, u32::MAX, HSP_BLOCKS).is_err());
        assert!(validate_hit_config(100, ceiling + 1, HSP_BLOCKS).is_err());
        assert!(validate_hit_config(0, 0, HSP_BLOCKS).is_err());
        assert!(validate_hit_config(100, 99, HSP_BLOCKS).is_err());
        assert!(validate_hit_config(100, 0, HSP_BLOCKS).is_err());
        assert!(validate_hit_config(100, 101, 0).is_err());
        assert_eq!(clamp_hit_capacity(100, 125, HSP_BLOCKS).unwrap(), 125);
        assert_eq!(
            clamp_hit_capacity(100, u32::MAX, HSP_BLOCKS).unwrap(),
            ceiling
        );
    }

    #[test]
    fn hit_buckets_match_the_plan_boundaries() {
        let mut s = HitStats::default();
        for n in [0, 1, 2, 4, 5, 32, 33, 256, 257, 1000] {
            s.observe(n);
        }
        assert_eq!(s.buckets, [1, 1, 2, 2, 2, 2]);
        assert_eq!(s.max, 1000);
        assert_eq!(s.seeds(), 10);
        assert_eq!(s.nonempty, 9);
        assert_eq!(s.total_hits, 1590);
        assert_eq!(
            s.quantile_nonempty(0.5),
            32,
            "middle of the 9 non-empty counts"
        );
        assert_eq!(s.quantile_nonempty(0.0), 1, "smallest non-empty count");
        assert_eq!(s.quantile_nonempty(1.0), 1000, "largest");
    }

    #[cfg(feature = "dense-anchors")]
    #[test]
    fn dense_compaction_preserves_order_and_clears_flags() {
        fn compact(flags: &mut [u8]) -> Vec<u32> {
            let counts: Vec<u32> = flags
                .chunks(SCAN_BLOCK as usize)
                .map(|block| block.iter().filter(|&&keep| keep != 0).count() as u32)
                .collect();
            let mut offsets = counts;
            let mut total = 0u32;
            for count in &mut offsets {
                let n = *count;
                *count = total;
                total += n;
            }
            let mut out = vec![0; total as usize];
            for (block, chunk) in flags.chunks_mut(SCAN_BLOCK as usize).enumerate() {
                let mut local = 0u32;
                for (lane, keep) in chunk.iter_mut().enumerate() {
                    if *keep != 0 {
                        out[(offsets[block] + local) as usize] =
                            (block * SCAN_BLOCK as usize + lane) as u32;
                        local += 1;
                    }
                    *keep = 0;
                }
            }
            out
        }

        for mut flags in [vec![0; 520], vec![1; 520], {
            let mut v = vec![0; 520];
            for i in [0, 31, 32, 255, 256, 519] {
                v[i] = 1;
            }
            v
        }] {
            let expected = flags
                .iter()
                .enumerate()
                .filter_map(|(i, &keep)| (keep != 0).then_some(i as u32))
                .collect::<Vec<_>>();
            assert_eq!(compact(&mut flags), expected);
            assert!(flags.iter().all(|&flag| flag == 0));
        }
    }

    #[cfg(all(feature = "dense-anchors", feature = "counters"))]
    #[test]
    fn dense_counters_keep_the_raw_hit_denominator() {
        let mut stats = super::HspStats::default();
        stats.observe_score_gate(10, 2);
        stats.observe(&[
            1 | (1 << 20) | (1 << 48),
            7 | (9 << 32),
            2 | (1 << 20),
            11 | (13 << 32),
        ]);

        assert_eq!(stats.hits, 10);
        assert_eq!(stats.score_gate_survivors, 2);
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.total[0], 8);
        assert_eq!(stats.total[2], 1);
        assert_eq!(stats.total[3], 1);
    }
}
