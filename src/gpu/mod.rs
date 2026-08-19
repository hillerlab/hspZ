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
//!   host; the scans themselves are device kernels.
//! * The engine is reference-scoped: `Engine::new` uploads the reference
//!   index and `swap_query` replaces only the query side, so one reference bin
//!   serves many query bins; the pass refuses to run on a never-swapped query.
//! * With the default async staging the kernel chain is
//!   enqueued without per-stage host waits and the seed upload rides a second
//!   stream, overlapping the previous batch's compute.
//!
//! Every stage is timed on the host clock so the phases add up to wall time;
//! kernel stages also carry their CUDA-event duration (`gpu ms`).

pub mod kernels;

use crate::hsp::SegmentPair;
use crate::seed::Shape;
use crate::timing::Phases;
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use kernels::{BLOCK_SIZE, HSP_BLOCKS, HSP_THREADS, MAX_BLOCKS, MAX_THREADS, SCAN_BLOCK};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `seed_filter.cu: MAX_HITS_PER_GB`.
const MAX_HITS_PER_GB: u64 = 4_194_304;

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
}

/// Hits-per-seed distribution for one chunk.
#[derive(Debug, Default, Clone)]
pub struct HitStats {
    /// Seeds with 0, 1, 2-4, 5-32, 33-256, >256 reference hits.
    pub buckets: [u64; 6],
    pub max: u32,
    pub total_hits: u64,
    pub nonempty: u64,
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
        self.max = self.max.max(n);
        self.total_hits += n as u64;
        if n > 0 {
            self.nonempty += 1;
            self.counts.push(n);
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
        self.max = self.max.max(other.max);
        self.total_hits += other.total_hits;
        self.nonempty += other.nonempty;
        self.counts.extend_from_slice(&other.counts);
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
        let mut out = String::from("  hits/seed      seeds        %\n");
        for (label, n) in HitStats::LABELS.iter().zip(self.buckets) {
            out.push_str(&format!(
                "  {label:<10} {n:>10} {:>7.2}%\n",
                n as f64 / seeds as f64 * 100.0
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
    noentropy: u32,
    max_hits: u32,
    timing: bool,
    /// `find_hsps` grid, resolved once from the config.
    hsp_blocks: u32,
    /// Keep the pre-dedup anchors in [`FilterOutput::raw`].
    pub dump_raw: bool,
    /// Accumulate the hits-per-seed distribution.
    pub collect_hit_stats: bool,
    /// Raw-anchor same-diagonal census (`HSPZ_ANCHOR_CENSUS`). Off the timed path.
    pub census: Option<crate::census::AnchorCensus>,
    pub phases: Phases,
    pub hit_stats: HitStats,
    #[cfg(feature = "counters")]
    pub hsp_stats: HspStats,
    #[cfg(feature = "counters")]
    pub groups: crate::hsp::Groups,
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
    /// Reuse `d_seeds` / `d_hit_num` across batches instead of allocating and
    /// freeing them per batch. Off by default; the earlier `d_hsp`/`d_done`
    /// result showed `cuMemFree` can cost more than the named allocation stage,
    /// so this is judged on whole runtime, not on alloc time.
    pub persistent_seed_buffers: bool,
    /// Two seed buffers, so batch *N+1*'s upload can be in flight while batch
    /// *N*'s kernels read the other one. Slot parity is `batch % 2`.
    seed_slots: [DeviceBuffer<u64>; 2],
    seed_len: [u32; 2],
    buf_hit_num: DeviceBuffer<u32>,
    seed_shape: DeviceBuffer<u32>,
    buf_seed_kmer: DeviceBuffer<u32>,
    /// Retained until `seed_and_filter` drains the stream; a per-call temporary
    /// could be freed while the async add-offset kernel still reads it.
    buf_seed_offsets: DeviceBuffer<u32>,
    /// Free list of pinned host staging buffers. Page-locking is
    /// expensive — `cuMemHostAlloc` of two 26 MB buffers cost ~13 ms per pass
    /// when this was done per pass — so they live for the engine's lifetime and
    /// are handed out and returned instead of reallocated.
    pinned_slots: Vec<cuda_core::PinnedHostBuffer<u64>>,
    /// Set once a pinned buffer has actually been allocated and returned, so a
    /// silent fallback to pageable memory cannot masquerade as a pinned run.
    pinned_ok: bool,
    /// The GPU-timeline gap between one stage's end event and the next stage's
    /// start event, summed. The phase table cannot recover this — it retains
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
    /// Per-pair totals. One dominant pair means a specific round trip worth
    /// removing; a flat spread over hundreds of pairs means launch and enqueue
    /// overhead, which is structural. The single largest gap cannot tell those
    /// apart. Linear scan over a handful of distinct stage pairs.
    gap_pairs: Vec<((&'static str, &'static str), f32, u64)>,
    /// Generate the query seed stream on the device instead of walking it on
    /// the host. Both paths are compiled; the executor sets this from the worker
    /// count because the trade reverses with GPU count — on one GPU the device
    /// seeder adds ~165 s of device work against a host tail that pinned async
    /// H->D already hides, and on two T4s it is worth -20.3% of wall because
    /// that tail is 397 s and exposed.
    pub device_seeds: bool,
    /// Device uploads of the reference index. One per `Engine` construction;
    /// the executor checks it against the reference-bin count.
    reference_uploads: u32,
    /// Query bins run against the resident reference bin.
    query_swaps: u32,
    /// Stages whose CUDA events are recorded but not yet read.
    ///
    /// Measuring a stage must not force it to complete, so `end_stage` records
    /// the end event and moves on; the durations are read at the next genuine
    /// host dependency, when the events are known to have completed.
    pending: Vec<PendingStage>,
    /// `stage_syncs` are the host waits that exist only to measure or to
    /// serialise stages that stream order already serialises. `pipeline_syncs`
    /// are the real dependencies: a host read of a device result, or a free that
    /// must not race a queued kernel.
    stage_syncs: u64,
    pipeline_syncs: u64,
    /// Drop the per-stage host synchronization (`--async-stages`). Off by
    /// default: the removal measured performance-neutral on an L4, so the
    /// shipped path keeps the simpler invariant. The async path needs this
    /// flag, and `parity.sh` keeps it honest.
    pub async_stages: bool,
    /// Upload seeds on a second stream instead of blocking the host.
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
        // Hazard 1: freeing memory a queued kernel still references is UB,
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
    /// `0` derives the KegAlign default from total device memory.
    pub max_hits: u32,
    pub timing: bool,
    /// `find_hsps` grid. `0` uses [`HSP_BLOCKS`], the ZLUDA optimum. Runtime
    /// rather than `const` so a grid sweep needs no rebuild.
    pub hsp_blocks: u32,
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

        let max_hits = resolve_max_hits(ctx, cfg.max_hits);

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
            noentropy: cfg.noentropy as u32,
            max_hits,
            timing: cfg.timing,
            hsp_blocks: if cfg.hsp_blocks > 0 {
                cfg.hsp_blocks
            } else {
                HSP_BLOCKS
            },
            dump_raw: false,
            collect_hit_stats: false,
            census: crate::census::AnchorCensus::enabled()
                .then(crate::census::AnchorCensus::default),
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
            persistent_seed_buffers: false,
            seed_slots: [unsafe { uninitialized(&stream, 0)? }, unsafe {
                uninitialized(&stream, 0)?
            }],
            seed_len: [0, 0],
            buf_hit_num: unsafe { uninitialized(&stream, 0)? },
            seed_shape: unsafe { uninitialized(&stream, 0)? },
            buf_seed_kmer: unsafe { uninitialized(&stream, 0)? },
            buf_seed_offsets: unsafe { uninitialized(&stream, 0)? },
            pinned_slots: Vec::new(),
            pinned_ok: false,
            last_end: None,
            last_name: "",
            gap_ms: 0.0,
            gap_n: 0,
            gap_max: 0.0,
            gap_max_pair: ("", ""),
            gap_pairs: Vec::new(),
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
        engine.stream.synchronize()?;
        engine.copy_stream.synchronize()?;
        phases.add("upload ref/query/tables", t.elapsed());
        Ok(engine)
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
        self.query_seq = DeviceBuffer::from_host(&self.stream, query_seq)?;
        self.query_rc_seq = DeviceBuffer::from_host(&self.stream, query_rc_seq)?;
        self.query_len = query_seq.len() as u32;
        self.query_swaps += 1;
        Ok(())
    }

    /// How many times this engine uploaded a reference index.
    ///
    /// The executor asserts this equals the *reference-bin* count, not the
    /// work-unit count. Counting host `SeedTable::build` calls alone would miss a
    /// refactor that rebuilt the `Engine` per pair, which is the regression
    /// this guards against — so it counts device uploads.
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

    /// The `find_hsps` grid in force.
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
    /// `--async-seed-copy`: the copy goes on a second stream, ordered
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
        let mut d_block_sums = DeviceBuffer::<u32>::zeroed(&self.stream, blocks as usize)?;
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
                &mut d_block_sums,
                span,
            )?;
        }
        self.end_stage(stage, "seed scan blocks")?;

        let t = Instant::now();
        let mut sums = d_block_sums.to_host_vec(&self.stream)?;
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
        self.buf_seed_offsets = DeviceBuffer::from_host(&self.stream, &sums)?;
        self.pipeline_syncs += 1;
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

    /// Byte-for-byte seed oracle used only by the device-seeds correctness
    /// build.
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
    /// already staged in `slot` by [`upload_seeds`](Self::upload_seeds).
    pub fn seed_and_filter(
        &mut self,
        slot: usize,
        rev: bool,
    ) -> Result<FilterOutput, Box<dyn std::error::Error>> {
        // Refuse to run against a query that was never swapped in.
        //
        // `Engine::new` is reference-only, so a fresh engine starts with
        // no query. Forgetting `swap_query` compiles cleanly — the fields were
        // removed from `EngineConfig`, not renamed, so nothing type-checks their
        // presence — and the whole pipeline would then run against an empty query
        // and emit zero HSPs with no error. That already happened once during the
        // refactor and only the oracle caught it.
        //
        // An `Err`, not a `debug_assert`: this is an internal invariant that must
        // hold in release builds, which is where a whole-genome run happens.
        if self.query_len == 0 {
            return Err("Engine has no query: swap_query() must be called before \
                 seed_and_filter()"
                .into());
        }
        let num_seeds = self.seed_len[slot];

        // Allocation and transfer are timed apart, because the
        // remedies are different — a grown-once buffer fixes one and does
        // nothing for the other.
        let t = Instant::now();
        // `uninitialized_async` would be the honest allocation-only probe, but
        // ZLUDA does not implement cuMemAllocAsync (DriverError 801), so the
        // zeroing memset is counted as part of allocation. That is the right
        // grouping for the decision anyway: a grown-once buffer removes both.
        // Both memsets are provably dead: `d_seeds` is fully
        // overwritten by the copy below, and `d_hit_num` by `find_num_hits`'
        // grid-stride loop over every `id < num_seeds`, before `scan_blocks`
        // reads it. Under ZLUDA this cut `alloc seeds + counts` 38.3 -> 31.3 ms
        // but left whole runtime inside noise, so production keeps the zeroing.
        //
        // The persistent option instead keeps both buffers across batches. The
        // two are deliberately separate: one changes initialization, the other
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
            // The rejected arm. The seed slots stay persistent regardless: an
            // upload may be in flight into one of them, and reallocating under a
            // queued copy is the use-after-free this guards.
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

        // The upload was issued a batch ago on the copy stream, so all that is
        // left is the ordering edge. `query()` never blocks and tells us whether
        // the overlap actually happened — a stall is the copy still in flight
        // when its compute wanted it, which is the honest measure of
        // "overlap > 0".
        if self.async_seed_copy && num_seeds > 0 {
            if !self.copy_ready[slot].query()? {
                self.seed_copy_stalls += 1;
            }
            self.stream.wait(&self.copy_ready[slot])?;
        }

        let stage = self.stage();
        // SAFETY: grid-stride kernel; `index_table` covers every seed value and
        // `d_hit_num` has one slot per seed.
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
        self.end_stage(stage, "find_num_hits")?;

        if self.collect_hit_stats {
            // Benchmark-only: the raw counts are otherwise never needed on the
            // host, so this full copy stays out of normal runs.
            let t = Instant::now();
            // A persistent buffer holds the high-water capacity, so only the
            // first `num_seeds` entries belong to this batch.
            let raw = self.buf_hit_num.to_host_vec(&self.stream)?;
            self.absorb_sync()?;
            for &n in &raw[..num_seeds as usize] {
                self.hit_stats.observe(n);
            }
            self.phases.add("hit-distribution stats", t.elapsed());
        }

        // The cumulative counts stay on the device. A block-local
        // scan plus an add-back turns them into the global inclusive scan, and
        // only the per-block sums (num_seeds/256 u32) cross the bus instead of
        // the whole array in each direction.
        let blocks = num_seeds.div_ceil(SCAN_BLOCK);
        let mut d_block_sums = DeviceBuffer::<u32>::zeroed(&self.stream, blocks as usize)?;
        let stage = self.stage();
        // SAFETY: one element per thread over `num_seeds`, one sum slot per block.
        unsafe {
            self.module.scan_blocks(
                &self.stream,
                LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (SCAN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut self.buf_hit_num,
                &mut d_block_sums,
                num_seeds,
            )?;
        }
        self.end_stage(stage, "scan_blocks")?;

        let t = Instant::now();
        // REQUIRED: the host computes the exclusive scan of the block sums and
        // the total decides the chunk walk, so this round trip is a real
        // dependency. Moving the reduction onto the device would remove it, but
        // that is a redesign, not a fix.
        let mut sums = d_block_sums.to_host_vec(&self.stream)?;
        self.absorb_sync()?;
        let mut acc = 0u32;
        for s in sums.iter_mut() {
            let total = *s;
            *s = acc;
            acc += total;
        }
        let num_hits = acc;
        let d_offsets = DeviceBuffer::from_host(&self.stream, &sums)?;
        self.pipeline_syncs += 1;
        self.phases.add("block-sum round trip", t.elapsed());

        let stage = self.stage();
        // SAFETY: same geometry as `scan_blocks`, so block `b` reads offset `b`.
        unsafe {
            self.module.add_block_offsets(
                &self.stream,
                LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (SCAN_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut self.buf_hit_num,
                &d_offsets,
                num_seeds,
            )?;
        }
        self.end_stage(stage, "add_block_offsets")?;

        let mut out = FilterOutput {
            hsps: Vec::new(),
            num_hits,
            raw_hsps: 0,
            raw: Vec::new(),
        };
        if num_hits == 0 {
            // `d_block_sums`/`d_offsets` are freed on the way out and
            // `add_block_offsets` may still be queued reading them (hazard 2).
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
            let cumulative = self.buf_hit_num.to_host_vec(&self.stream)?;
            self.absorb_sync()?;
            chunk_limits(&cumulative[..num_seeds as usize], self.max_hits)
        };
        self.phases.add("chunk prep (lower_bound)", t.elapsed());

        for (start_seed_index, limit_pos, start_hit_val, end_hit_val) in chunks {
            let iter_num_seeds = limit_pos + 1 - start_seed_index;
            let iter_num_hits = end_hit_val - start_hit_val;
            if iter_num_hits == 0 {
                continue;
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
                reserve(
                    &mut self.buf_anchor,
                    &self.stream,
                    iter_num_hits as usize,
                    &mut self.pipeline_syncs,
                )?;
                reserve(
                    &mut self.buf_flags,
                    &self.stream,
                    iter_num_hits as usize,
                    &mut self.pipeline_syncs,
                )?;
            }
            self.phases.add("alloc hit buffers", t.elapsed());

            let stage = self.stage();
            // SAFETY: one thread per seed; every store lands inside
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
            unsafe {
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
            self.end_stage(stage, "find_hits")?;

            if self.census.is_some() {
                #[cfg(feature = "dense-anchors")]
                {
                    let host = self.buf_anchor.to_host_vec(&self.stream)?;
                    self.absorb_sync()?;
                    if let Some(c) = self.census.as_mut() {
                        c.ingest(&host[..iter_num_hits as usize]);
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
                let stage = self.stage();
                // SAFETY: one warp per raw hit; every active flag is written.
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
                        self.hspthresh,
                        iter_num_hits,
                        &self.buf_anchor,
                        &mut self.buf_flags,
                    )?;
                }
                self.end_stage(stage, "score_gate")?;

                let blocks = iter_num_hits.div_ceil(SCAN_BLOCK);
                // SAFETY: `count_survivors` writes one sum for every launched
                // block before the host copy reads it.
                let mut d_sums = unsafe { uninitialized::<u32>(&self.stream, blocks as usize)? };
                let stage = self.stage();
                unsafe {
                    self.module.count_survivors(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: (blocks, 1, 1),
                            block_dim: (SCAN_BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &self.buf_flags,
                        &mut d_sums,
                        iter_num_hits,
                    )?;
                }
                self.end_stage(stage, "count_survivors")?;

                let t = Instant::now();
                let mut sums = d_sums.to_host_vec(&self.stream)?;
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

                let d_offsets = DeviceBuffer::from_host(&self.stream, &sums)?;
                self.pipeline_syncs += 1;
                let stage = self.stage();
                unsafe {
                    self.module.emit_survivors(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: (blocks, 1, 1),
                            block_dim: (SCAN_BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut self.buf_flags,
                        &d_offsets,
                        &mut self.buf_survivors,
                        iter_num_hits,
                    )?;
                }
                self.end_stage(stage, "emit_survivors")?;
                (total, d_offsets)
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
            let mut d_stats = DeviceBuffer::<u64>::zeroed(&self.stream, stats_len)?;
            if !self.async_stages {
                self.stream.synchronize()?;
                self.stage_syncs += 1;
            }
            self.phases.add("alloc materializer buffers", t.elapsed());
            let (free, total) = device_memory();
            self.peak_used = self.peak_used.max(total.saturating_sub(free));

            #[cfg(feature = "counters")]
            self.hsp_stats.set_strand(rev);
            let query = if rev {
                &self.query_rc_seq
            } else {
                &self.query_seq
            };
            #[cfg(feature = "dense-anchors")]
            let anchor_ptr = self.buf_anchor.cu_deviceptr() as *const u64;
            #[cfg(not(feature = "dense-anchors"))]
            let anchor_ptr = core::ptr::null::<u64>();
            #[cfg(feature = "dense-anchors")]
            let survivor_ptr = self.buf_survivors.cu_deviceptr() as *const u32;
            #[cfg(not(feature = "dense-anchors"))]
            let survivor_ptr = core::ptr::null::<u32>();
            let stage = self.stage();
            // SAFETY: one warp per work item; `hsp`/`done` are sized to the
            // materializer count and the dense pointers cover every emitted ID.
            unsafe {
                self.module.find_hsps(
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
                    self.noentropy,
                    self.xdrop,
                    self.hspthresh,
                    materializer_hits,
                    LOG4,
                    anchor_ptr,
                    survivor_ptr,
                    &mut self.buf_hsp,
                    &mut self.buf_done,
                    &mut d_stats,
                )?;
            }
            self.end_stage(stage, "find_hsps")?;

            #[cfg(feature = "counters")]
            {
                let raw = d_stats.to_host_vec(&self.stream)?;
                self.absorb_sync()?;
                self.hsp_stats.observe(&raw);
            }
            #[cfg(feature = "counters")]
            let _ = &d_stats;

            // Scan the done flags in place on the device with the
            // same two-kernel machinery already accepted for the hit counts, so
            // only the per-block sums (`materializer_hits/256` u32) cross the bus
            // instead of the whole flag array in each direction. `num_anchors`
            // falls out of the block-sum scan, and `compress_output` is
            // unchanged — it still reads a full inclusive scan.
            let done_blocks = materializer_hits.div_ceil(SCAN_BLOCK);
            let mut d_done_sums = DeviceBuffer::<u32>::zeroed(&self.stream, done_blocks as usize)?;
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
                    &mut d_done_sums,
                    materializer_hits,
                )?;
            }
            self.end_stage(stage, "scan_blocks (done)")?;

            let t = Instant::now();
            // REQUIRED: `num_anchors` sizes the reduced buffer and decides whether
            // this chunk emits anything at all.
            let mut sums = d_done_sums.to_host_vec(&self.stream)?;
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

            let d_offsets = DeviceBuffer::from_host(&self.stream, &sums)?;
            self.pipeline_syncs += 1;
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
                    &d_offsets,
                    materializer_hits,
                )?;
            }
            self.end_stage(stage, "add_block_offsets (done)")?;

            let t = Instant::now();
            let mut d_reduced =
                DeviceBuffer::<SegmentPair>::zeroed(&self.stream, num_anchors as usize)?;
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
                    &mut d_reduced,
                    materializer_hits,
                )?;
            }
            self.end_stage(stage, "compress_output")?;

            let t = Instant::now();
            // REQUIRED: this is the output. It also drains the stream, which is
            // what makes the per-chunk temporaries safe to drop below.
            let mut anchors = d_reduced.to_host_vec(&self.stream)?;
            self.absorb_sync()?;
            self.phases.add("D->H anchors", t.elapsed());

            if self.dump_raw {
                out.raw.extend_from_slice(&anchors);
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
            // Pair the previous stage's end with this stage's start. Both are
            // on one stream, so the delta is GPU-timeline idle — the bubble the host put
            // there. A negative or absurd reading would mean the events are unordered,
            // so anything outside [0, 1000] ms is dropped rather than trusted.
            if let (Some(prev), Some((start, _))) = (self.last_end.as_ref(), p.events.as_ref())
                && let Ok(gap) = prev.elapsed_ms(start)
                && (0.0f32..=1000.0f32).contains(&gap)
            {
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
            }
            if let Some((_, end)) = p.events {
                self.last_end = Some(end);
                self.last_name = p.name;
            }
            self.phases.add_gpu(p.name, p.host, gpu_ms);
        }
        Ok(())
    }

    /// Clear the gap accumulator so a reading covers one timed pass only.
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
        self.last_end = None;
        self.last_name = "";
    }

    /// Per-pair gap totals, largest first.
    pub fn gap_pairs(&self) -> Vec<((&'static str, &'static str), f32, u64)> {
        let mut v = self.gap_pairs.clone();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// `(summed gap ms, pairs measured, largest gap, its stage pair)`.
    pub fn stage_gaps(&self) -> (f32, u64, f32, (&'static str, &'static str)) {
        (self.gap_ms, self.gap_n, self.gap_max, self.gap_max_pair)
    }

    /// A genuine host dependency: the host is about to read a device
    /// result, free a buffer a queued kernel touches, or return to the caller.
    fn sync_pipeline(&mut self) -> Result<(), DriverError> {
        self.stream.synchronize()?;
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
    /// stages. This must fall to zero when the waits are removed.
    pub fn stage_syncs(&self) -> u64 {
        self.stage_syncs
    }

    /// Uploads issued, and uploads that had not completed when their compute
    /// needed them. `stalls == 0` means every DMA hid behind compute.
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
/// of per reference bin — is invisible in output and only shows up as ~1 GB of
/// extra H->D traffic per avoided reuse.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Lifecycle {
    pub seed_table_builds: u32,
    pub engine_creations: u32,
    pub reference_uploads: u32,
    pub query_swaps: u32,
    pub work_units_executed: u32,
}

impl Lifecycle {
    /// Checks the reference-scoped invariants against a plan's shape.
    ///
    /// Returns the first violation as a message rather than panicking, so a
    /// caller can report it alongside the rest of a profile.
    pub fn check(&self, reference_bins: u32, work_units: u32) -> Result<(), String> {
        let want = [
            ("seed_table_builds", self.seed_table_builds, reference_bins),
            ("engine_creations", self.engine_creations, reference_bins),
            ("reference_uploads", self.reference_uploads, reference_bins),
            ("query_swaps", self.query_swaps, work_units),
            ("work_units_executed", self.work_units_executed, work_units),
        ];
        for (name, got, expect) in want {
            if got != expect {
                return Err(format!(
                    "lifecycle: {name} = {got}, expected {expect} \
                     (reference_bins {reference_bins}, work_units {work_units})"
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
    Ok(unsafe { DeviceBuffer::from_raw_parts(ptr, len, ctx) })
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

/// Splits the scanned hit counts into `(start_seed_index, limit_pos,
/// start_hit_val)` chunks, reproducing the `lower_bound` walk in
/// `SeedAndFilter`.
fn chunk_limits(hit_num: &[u32], max_hits: u32) -> Vec<(u32, u32, u32, u32)> {
    let num_seeds = hit_num.len() as u32;
    let num_hits = *hit_num.last().unwrap_or(&0);
    let num_iter = num_hits / max_hits + 1;

    let mut limits = Vec::with_capacity(num_iter as usize);
    let mut iter_hit_limit = max_hits;
    for _ in 0..num_iter - 1 {
        let pos = hit_num.partition_point(|&v| v < iter_hit_limit) as u32 - 1;
        iter_hit_limit = hit_num[pos as usize] + max_hits;
        limits.push(pos);
    }
    limits.push(num_seeds - 1);

    let mut out = Vec::with_capacity(limits.len());
    let mut start_seed_index = 0;
    let mut start_hit_val = 0;
    for pos in limits {
        out.push((start_seed_index, pos, start_hit_val, hit_num[pos as usize]));
        start_seed_index = pos + 1;
        start_hit_val = hit_num[pos as usize];
    }
    out
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

/// `max_hits` in force: `requested`, or the device-derived default when `0`.
/// Public so the preflight can size its budget before any `Engine` exists.
pub fn resolve_max_hits(ctx: &Arc<CudaContext>, requested: u32) -> u32 {
    if requested > 0 {
        requested
    } else {
        default_max_hits(ctx)
    }
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
    #[cfg(feature = "dense-anchors")]
    use super::SCAN_BLOCK;
    use super::{HitStats, Lifecycle, chunk_limits};

    /// The counters exist to catch a reference index rebuilt per work unit
    /// instead of per reference bin — a regression that changes no output and
    /// shows up only as ~1 GB of extra H->D traffic per work unit.
    #[test]
    fn lifecycle_rejects_per_work_unit_reference_rebuilds() {
        // A 2x3 plan: 2 reference bins, 6 work units.
        let good = Lifecycle {
            seed_table_builds: 2,
            engine_creations: 2,
            reference_uploads: 2,
            query_swaps: 6,
            work_units_executed: 6,
        };
        assert!(good.check(2, 6).is_ok(), "{:?}", good.check(2, 6));

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
    fn single_chunk_when_hits_fit() {
        let hit_num = vec![2, 5, 9];
        assert_eq!(chunk_limits(&hit_num, 1000), vec![(0, 2, 0, 9)]);
    }

    #[test]
    fn chunk_boundaries_follow_the_lower_bound_walk() {
        // 4 seeds, running totals 3/7/11/15, MAX_HITS = 5.
        let hit_num = vec![3, 7, 11, 15];
        let chunks = chunk_limits(&hit_num, 5);
        assert_eq!(chunks.len(), 4);
        assert_eq!(
            chunks[0],
            (0, 0, 0, 3),
            "first chunk stops before overshooting 5"
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
