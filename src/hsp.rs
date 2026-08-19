// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! The HSP record and the ordering / dedup rules that decide the final output.
//!
//! Port of the `segmentPair` struct plus the `hspComp` / `hspEqual` /
//! `hspCompLastz` functors in `seed_filter.cu`, and of the `.segments` writer in
//! `segment_printer.cpp`.

use crate::sequence::{Chr, chr_at};
use std::cmp::Ordering;

/// `parameters.h: segmentPair`. 16 bytes, POD, laid out for the GPU as-is.
///
/// `ref_start` / `query_start` are block-relative offsets of the first base of
/// the segment; `len` is the extent, so the segment covers `start ..= start+len`
/// inclusive once printed.
/// `align(16)` is a codegen lever, not a layout change: same 16 bytes, same
/// field offsets, and device buffers are 256-byte aligned so element `i` sits on
/// a 16-byte boundary either way. Declaring it lets LLVM emit one
/// `st.global.v4.b32` instead of four `st.global.b32` — in `find_hits`' per-hit
/// write and, worth more, in `find_hsps`' HSP store.
///
/// Measured on an NVIDIA L4: `find_hsps` -4.25% (A) / -2.49% (B), whole run
/// -3.3% / -3.2% drift-corrected. Under ZLUDA the same change was flat, so this
/// is free there and a real win on native NVIDIA.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, cuda_core::DeviceCopy)]
pub struct SegmentPair {
    pub ref_start: u32,
    pub query_start: u32,
    pub len: u32,
    pub score: i32,
}

const _: () = assert!(size_of::<SegmentPair>() == 16);
const _: () = assert!(align_of::<SegmentPair>() == 16);

impl SegmentPair {
    /// The anti-diagonal the segment lies on. KegAlign computes this with
    /// wrapping `uint32_t` arithmetic and compares it unsigned, so segments
    /// whose query start exceeds their reference start land at the top of the
    /// order rather than the bottom.
    #[inline]
    pub fn diagonal(&self) -> u32 {
        self.ref_start.wrapping_sub(self.query_start)
    }

    #[inline]
    fn ref_end(&self) -> u32 {
        self.ref_start.wrapping_add(self.len)
    }
}

/// `hspComp`: diagonal, then reference start, then length ascending, then score
/// descending. Used to bring duplicates and contained segments next to each
/// other before dedup.
pub fn by_diagonal(x: &SegmentPair, y: &SegmentPair) -> Ordering {
    x.diagonal()
        .cmp(&y.diagonal())
        .then(x.ref_start.cmp(&y.ref_start))
        .then(x.len.cmp(&y.len))
        .then(y.score.cmp(&x.score))
}

/// `hspCompLastz`: the order LASTZ expects in a `.segments` file.
pub fn by_lastz(x: &SegmentPair, y: &SegmentPair) -> Ordering {
    x.query_start
        .cmp(&y.query_start)
        .then(x.ref_start.cmp(&y.ref_start))
        .then(x.len.cmp(&y.len))
        .then(y.score.cmp(&x.score))
}

/// `hspEqual`: same diagonal and one segment contains the other. Note this is
/// *not* transitive — two disjoint segments can both sit inside a third — which
/// is why dedup has to compare against the previous input element rather than
/// the last kept one.
pub fn contained(x: &SegmentPair, y: &SegmentPair) -> bool {
    x.diagonal() == y.diagonal()
        && ((x.ref_start >= y.ref_start && x.ref_end() <= y.ref_end())
            || (y.ref_start >= x.ref_start && y.ref_end() <= x.ref_end()))
}

/// The `stable_sort -> unique_copy -> stable_sort` tail of `SeedAndFilter`.
///
/// `thrust::unique_copy` keeps the first element of every run of adjacent
/// equivalent elements, deciding with `pred(input[i-1], input[i])` — against the
/// previous *input* element, not the previously kept one.
pub fn dedup_and_order(hsps: &mut Vec<SegmentPair>, mut groups: Option<&mut Groups>) {
    hsps.sort_by(by_diagonal);
    let mut kept: Vec<SegmentPair> = Vec::with_capacity(hsps.len());
    for (i, h) in hsps.iter().enumerate() {
        if i == 0 || !contained(&hsps[i - 1], h) {
            kept.push(*h);
            if let Some(g) = groups.as_deref_mut() {
                g.sizes.push(1);
            }
        } else if let Some(g) = groups.as_deref_mut() {
            // Removed: attributed to the run head that survives it. The
            // production predicate decides this, not a geometric look-alike.
            *g.sizes
                .last_mut()
                .expect("a survivor precedes every removal") += 1;
            let prev = &hsps[i - 1];
            if prev.ref_start == h.ref_start
                && prev.query_start == h.query_start
                && prev.len == h.len
            {
                g.duplicate += 1;
            } else {
                g.contained += 1;
            }
            // `contained()` requires equal diagonals, so every removal is by
            // construction same-diagonal; counted to make that explicit.
            debug_assert_eq!(prev.diagonal(), h.diagonal());
            g.same_diagonal += 1;
        }
    }
    kept.sort_by(by_lastz);
    *hsps = kept;
}

/// Raw-HSP provenance through the production dedup.
#[derive(Debug, Default, Clone)]
pub struct Groups {
    /// Raw HSPs collapsing into each surviving HSP, one entry per survivor.
    pub sizes: Vec<u32>,
    pub duplicate: u64,
    pub contained: u64,
    pub same_diagonal: u64,
}

impl Groups {
    /// Renders the raw-HSP provenance table for the `counters` feature.
    #[cfg(feature = "counters")]
    pub fn report(&mut self) -> String {
        let raw: u64 = self.sizes.iter().map(|&s| s as u64).sum();
        let finals = self.sizes.len().max(1) as u64;
        let q = |v: &mut Vec<u32>, p: f64| -> u32 {
            if v.is_empty() {
                return 0;
            }
            let k = (((v.len() - 1) as f64) * p) as usize;
            *v.select_nth_unstable(k).1
        };
        let removed = self.duplicate + self.contained;
        format!(
            "  raw HSPs {raw}  final HSPs {finals}\n  raw per final: mean {:.3} median {} p95 {} \
             p99 {} max {}\n  removed {removed} ({:.2}% of raw): duplicate {} ({:.2}%), contained \
             {} ({:.2}%)\n  removals on the same diagonal as their survivor: {} ({:.2}%)\n",
            raw as f64 / finals as f64,
            q(&mut self.sizes, 0.5),
            q(&mut self.sizes, 0.95),
            q(&mut self.sizes, 0.99),
            self.sizes.iter().copied().max().unwrap_or(0),
            removed as f64 / raw.max(1) as f64 * 100.0,
            self.duplicate,
            self.duplicate as f64 / raw.max(1) as f64 * 100.0,
            self.contained,
            self.contained as f64 / raw.max(1) as f64 * 100.0,
            self.same_diagonal,
            self.same_diagonal as f64 / removed.max(1) as f64 * 100.0,
        )
    }
}

/// One printed `.segments` record, still numeric.
///
/// This is the unit `-D` partitions: partition structs, never text.
/// Coordinates are exactly what gets printed — 1-based, chromosome-relative,
/// inclusive at both ends — because the oracle's partitioner reads them off the
/// printed file and its diagonal keys are defined on those values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    pub r_chr: u32,
    pub q_chr: u32,
    pub r_start: usize,
    pub r_end: usize,
    pub q_start: usize,
    pub q_end: usize,
    pub score: i32,
}

impl Record {
    /// The oracle's midpoints: `half_dist = (seq1_end - seq1_start) / 2` is
    /// computed once from the *reference* span and added to both starts. That is
    /// not a typo in the script — an HSP has the same length on both sequences,
    /// so the two spans agree and the shared `half_dist` is exact.
    fn mids(&self) -> (i64, i64) {
        let half = ((self.r_end - self.r_start) / 2) as i64;
        (self.r_start as i64 + half, self.q_start as i64 + half)
    }

    /// Diagonal sort key. Plus strand sorts by the *sum*, minus by the
    /// *difference* — verified against `diagonal_partition.py`, whose own
    /// comments have the two branches labelled the wrong way round.
    pub fn diagonal_key(&self, strand: char) -> (i64, i64) {
        let (r_mid, q_mid) = self.mids();
        if strand == '-' {
            (q_mid - r_mid, r_mid)
        } else {
            (q_mid + r_mid, r_mid)
        }
    }
}

/// The records [`render_segments`] would print, in printed order.
pub fn records(hsps: &[SegmentPair], r_chrs: &[Chr], q_chrs: &[Chr], strand: char) -> Vec<Record> {
    let ordered: Box<dyn Iterator<Item = &SegmentPair>> = if strand == '-' {
        Box::new(hsps.iter().rev())
    } else {
        Box::new(hsps.iter())
    };
    ordered
        .map(|e| {
            let ri = chr_at(r_chrs, e.ref_start as usize);
            let qi = chr_at(q_chrs, e.query_start as usize);
            Record {
                r_chr: ri as u32,
                q_chr: qi as u32,
                r_start: e.ref_start as usize + 1 - r_chrs[ri].start,
                r_end: e.ref_start as usize + e.len as usize + 1 - r_chrs[ri].start,
                q_start: e.query_start as usize + 1 - q_chrs[qi].start,
                q_end: e.query_start as usize + e.len as usize + 1 - q_chrs[qi].start,
                score: e.score,
            }
        })
        .collect()
}

/// Renders records back to `.segments` text.
pub fn render_records(recs: &[Record], r_chrs: &[Chr], q_chrs: &[Chr], strand: char) -> String {
    let mut out = String::with_capacity(recs.len() * 64);
    for r in recs {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r_chrs[r.r_chr as usize].name,
            r.r_start,
            r.r_end,
            q_chrs[r.q_chr as usize].name,
            r.q_start,
            r.q_end,
            strand,
            r.score,
        ));
    }
    out
}

/// The exact text [`write_segments`] would write. Split out so a benchmark can
/// hash the output without touching the filesystem.
///
/// # Example
/// ```
/// let empty = render_segments(&[], &[], &[], '+');
/// assert_eq!(empty, "");
/// ```
pub fn render_segments(
    hsps: &[SegmentPair],
    r_chrs: &[Chr],
    q_chrs: &[Chr],
    strand: char,
) -> String {
    let ordered: Box<dyn Iterator<Item = &SegmentPair>> = if strand == '-' {
        Box::new(hsps.iter().rev())
    } else {
        Box::new(hsps.iter())
    };

    let mut out = String::with_capacity(hsps.len() * 64);
    for e in ordered {
        let r = &r_chrs[chr_at(r_chrs, e.ref_start as usize)];
        let q = &q_chrs[chr_at(q_chrs, e.query_start as usize)];
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.name,
            e.ref_start as usize + 1 - r.start,
            e.ref_start as usize + e.len as usize + 1 - r.start,
            q.name,
            e.query_start as usize + 1 - q.start,
            e.query_start as usize + e.len as usize + 1 - q.start,
            strand,
            e.score,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp(r: u32, q: u32, len: u32, score: i32) -> SegmentPair {
        SegmentPair {
            ref_start: r,
            query_start: q,
            len,
            score,
        }
    }

    #[test]
    fn diagonal_wraps_like_uint32() {
        assert_eq!(sp(10, 4, 0, 0).diagonal(), 6);
        assert_eq!(sp(4, 10, 0, 0).diagonal(), u32::MAX - 5);
        assert!(by_diagonal(&sp(10, 4, 0, 0), &sp(4, 10, 0, 0)) == Ordering::Less);
    }

    #[test]
    fn sort_key_is_diag_start_len_then_score_descending() {
        let mut v = vec![
            sp(5, 0, 3, 100),
            sp(5, 0, 3, 900),
            sp(5, 0, 1, 10),
            sp(4, 0, 9, 1),
        ];
        v.sort_by(by_diagonal);
        assert_eq!(
            v,
            vec![
                sp(4, 0, 9, 1),
                sp(5, 0, 1, 10),
                sp(5, 0, 3, 900),
                sp(5, 0, 3, 100)
            ]
        );
    }

    #[test]
    fn dedup_drops_contained_segments_on_the_same_diagonal() {
        // Same diagonal: [0,10) swallows [0,5) and [6,8); a different diagonal
        // is never merged even when the intervals nest.
        let mut v = vec![
            sp(0, 0, 10, 50),
            sp(0, 0, 5, 40),
            sp(6, 6, 2, 30),
            sp(0, 1, 4, 20),
        ];
        dedup_and_order(&mut v, None);
        assert_eq!(v.len(), 2, "one survivor per diagonal, got {v:?}");
        assert!(v.contains(&sp(0, 1, 4, 20)));
        assert!(
            v.contains(&sp(0, 0, 5, 40)),
            "shortest sorts first and is the one kept"
        );
    }

    #[test]
    fn dedup_compares_against_previous_input_not_last_kept() {
        // a=[0,5) b=[0,10) c=[6,8), all on diagonal 0. Comparing against the
        // previous input drops c (contained in b); comparing against the last
        // kept element (a) would have retained it.
        let mut v = vec![sp(0, 0, 5, 10), sp(0, 0, 10, 20), sp(6, 6, 2, 30)];
        dedup_and_order(&mut v, None);
        assert_eq!(v, vec![sp(0, 0, 5, 10)]);
    }

    #[test]
    fn lastz_order_is_query_major() {
        let mut v = vec![sp(9, 5, 1, 1), sp(1, 2, 1, 1), sp(0, 5, 1, 1)];
        v.sort_by(by_lastz);
        assert_eq!(v, vec![sp(1, 2, 1, 1), sp(0, 5, 1, 1), sp(9, 5, 1, 1)]);
    }
}
