// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! Raw-anchor same-diagonal neighbour census — an env-gated diagnostic
//! (`HSPZ_ANCHOR_CENSUS`), off the timed path.
//!
//! Neighbour-distance p50 ≈ 800 kb on *evaluated extension intervals*. This
//! census walks every packed seed-hit anchor and asks how many could be skipped
//! if a same-diagonal neighbour within 96 bases were treated as redundant.

/// Packed key: `(diagonal as u32) << 32 | ref_start`.
fn key(anchor: u64) -> u64 {
    let r = anchor as u32;
    let q = (anchor >> 32) as u32;
    (r.wrapping_sub(q) as u64) << 32 | r as u64
}

#[derive(Default)]
pub struct AnchorCensus {
    keys: Vec<u64>,
    chunks: u64,
    chunk_hits: u64,
    chunk_skippable: u64,
}

impl AnchorCensus {
    pub fn enabled() -> bool {
        std::env::var_os("HSPZ_ANCHOR_CENSUS").is_some()
    }

    pub fn ingest(&mut self, anchors: &[u64]) {
        if anchors.is_empty() {
            return;
        }
        let start = self.keys.len();
        self.keys.extend(anchors.iter().copied().map(key));
        let chunk = &mut self.keys[start..];
        chunk.sort_unstable();
        self.chunk_hits += chunk.len() as u64;
        self.chunk_skippable += skippable(chunk, 96);
        self.chunks += 1;
    }

    pub fn report(&mut self) -> String {
        self.keys.sort_unstable();
        let global = summarize(&self.keys);
        let chunk_frac = if self.chunk_hits == 0 {
            0.0
        } else {
            self.chunk_skippable as f64 / self.chunk_hits as f64 * 100.0
        };
        format!(
            "ANCHOR NEIGHBOUR CENSUS  (HSPZ_ANCHOR_CENSUS)\n\
             hits {}  chunks {}  diagonals {}\n\
             isolated hits {:.2}%   hits with any same-diag neighbour {:.2}%\n\
             neighbour-pair distance  p10 {}  p50 {}  p95 {}  p99 {}\n\
             pair gaps  ≤1 {:.2}%  ≤8 {:.2}%  ≤32 {:.2}%  ≤96 {:.2}%  ≤256 {:.2}%  ≤1024 {:.2}%\n\
             skippable if keep-first in a ≤96 cluster:  global {:.2}%  per-chunk {:.2}%\n",
            global.hits,
            self.chunks,
            global.diags,
            global.isolated_frac * 100.0,
            (1.0 - global.isolated_frac) * 100.0,
            global.p10,
            global.p50,
            global.p95,
            global.p99,
            global.le1,
            global.le8,
            global.le32,
            global.le96,
            global.le256,
            global.le1024,
            global.skippable_frac * 100.0,
            chunk_frac,
        )
    }
}

struct Summary {
    hits: u64,
    diags: u64,
    isolated_frac: f64,
    skippable_frac: f64,
    p10: u64,
    p50: u64,
    p95: u64,
    p99: u64,
    le1: f64,
    le8: f64,
    le32: f64,
    le96: f64,
    le256: f64,
    le1024: f64,
}

fn skippable(sorted: &[u64], window: u32) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let mut skip = 0u64;
    let mut i = 0usize;
    while i < sorted.len() {
        let d = sorted[i] >> 32;
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] >> 32 == d {
            j += 1;
        }
        let mut cluster = 1u64;
        let mut prev = sorted[i] as u32;
        // the index is the bucket id the histogram is keyed on
        #[allow(clippy::needless_range_loop)]
        for k in i + 1..j {
            let r = sorted[k] as u32;
            if r.saturating_sub(prev) <= window {
                cluster += 1;
            } else {
                skip += cluster.saturating_sub(1);
                cluster = 1;
            }
            prev = r;
        }
        skip += cluster.saturating_sub(1);
        i = j;
    }
    skip
}

fn summarize(sorted: &[u64]) -> Summary {
    let hits = sorted.len() as u64;
    let mut diags = 0u64;
    let mut isolated = 0u64;
    let mut gaps: Vec<u32> = Vec::new();
    let mut i = 0usize;
    while i < sorted.len() {
        let d = sorted[i] >> 32;
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] >> 32 == d {
            j += 1;
        }
        diags += 1;
        if j - i == 1 {
            isolated += 1;
        }
        for k in i + 1..j {
            gaps.push((sorted[k] as u32).saturating_sub(sorted[k - 1] as u32));
        }
        i = j;
    }
    let n = gaps.len() as f64;
    let pct = |w: u32| {
        if n == 0.0 {
            0.0
        } else {
            gaps.iter().filter(|&&g| g <= w).count() as f64 / n * 100.0
        }
    };
    let (le1, le8, le32, le96, le256, le1024) =
        (pct(1), pct(8), pct(32), pct(96), pct(256), pct(1024));
    let quantile = |gaps: &mut [u32], p: f64| -> u64 {
        if gaps.is_empty() {
            return 0;
        }
        let k = (((gaps.len() - 1) as f64) * p) as usize;
        *gaps.select_nth_unstable(k).1 as u64
    };
    Summary {
        hits,
        diags,
        isolated_frac: if hits == 0 {
            0.0
        } else {
            isolated as f64 / hits as f64
        },
        skippable_frac: if hits == 0 {
            0.0
        } else {
            skippable(sorted, 96) as f64 / hits as f64
        },
        p10: quantile(&mut gaps, 0.10),
        p50: quantile(&mut gaps, 0.50),
        p95: quantile(&mut gaps, 0.95),
        p99: quantile(&mut gaps, 0.99),
        le1,
        le8,
        le32,
        le96,
        le256,
        le1024,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anc(r: u32, q: u32) -> u64 {
        r as u64 | ((q as u64) << 32)
    }

    #[test]
    fn isolated_hits_are_not_skippable() {
        // Three different diagonals (r - q).
        let a = [anc(10, 0), anc(20, 5), anc(30, 1)];
        let mut keys: Vec<u64> = a.iter().copied().map(key).collect();
        keys.sort_unstable();
        assert_eq!(skippable(&keys, 96), 0);
        let s = summarize(&keys);
        assert_eq!(s.isolated_frac, 1.0);
        assert_eq!(s.skippable_frac, 0.0);
    }

    #[test]
    fn close_run_on_one_diagonal_skips_all_but_first() {
        // Same diagonal (r - q = 10), refs 100, 101, 150, 400.
        let a = [anc(100, 90), anc(101, 91), anc(150, 140), anc(400, 390)];
        let mut keys: Vec<u64> = a.iter().copied().map(key).collect();
        keys.sort_unstable();
        // clusters: [100,101,150] and [400] → skip 2
        assert_eq!(skippable(&keys, 96), 2);
        let s = summarize(&keys);
        assert!((s.le96 - 2.0 / 3.0 * 100.0).abs() < 0.01);
    }
}
