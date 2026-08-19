// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! `--diagonal-partition` — the port of KegAlign's `diagonal_partition.py`.
//!
//! The script partitions an already-written `.segments` file: it reads the text
//! back, parses coordinates, sorts, and rewrites the pieces. This does the same
//! partitioning on the numeric [`Record`]s before anything is formatted
//! so no unsplit file is ever created.
//!
//! Two deliberate differences from the script:
//!
//! * **Sizes are exact.** The script estimates a file's line count as
//!   `file_size / first_line_size`, because bytes are all it has. hspz knows the
//!   HSP count, so thresholds use counts directly. On files whose lines vary in
//!   width — every real one, since coordinates and scores differ in digits — the
//!   estimate is only approximate, so hspz's partition boundaries can differ
//!   from the script's on the same data. The partition *union* is unaffected.
//! * **Split order is deterministic.** The script iterates
//!   `data.keys() - skip_pairs`, a Python set difference, so the order in which
//!   pairs get `.split` numbers is unspecified. This keeps first-appearance
//!   order.

use crate::hsp::Record;

/// Files with fewer lines than this are never partitioned.
pub const MIN_CHUNK_SIZE: usize = 5000;
/// Cap on an estimated chunk size.
pub const MAX_CHUNK_SIZE: usize = 50000;

/// Prior logical output sizes, in HSPs, used to estimate a chunk size the way
/// the script estimates one from the sizes of `.segments` files already on disk.
#[derive(Debug, Default)]
pub struct Partitioner {
    history: Vec<usize>,
}

/// What to do with one logical output file.
#[derive(Debug, PartialEq, Eq)]
pub enum Plan {
    /// Emit a single unsplit `.segments`, as today.
    Whole(Vec<Record>),
    /// Emit `.split1 …` in this order; each inner vector is one file's records.
    Split(Vec<Vec<Record>>),
}

impl Partitioner {
    /// Python's `statistics.quantiles(data)` with the default `n=4` and the
    /// default *exclusive* method, returning `[Q1, Q2, Q3]`.
    ///
    /// Reimplemented rather than approximated because the chosen chunk size
    /// feeds file boundaries: `statistics.quantiles` interpolates at position
    /// `k * (n + 1) / 4` over the sorted sample.
    fn quantiles_exclusive(sorted: &[usize]) -> Option<[f64; 3]> {
        let n = sorted.len();
        if n < 2 {
            return None;
        }
        let at = |k: usize| -> f64 {
            let pos = k as f64 * (n + 1) as f64 / 4.0;
            // Clamp into the sample, matching CPython's edge handling.
            if pos < 1.0 {
                return sorted[0] as f64;
            }
            if pos >= n as f64 {
                return sorted[n - 1] as f64;
            }
            let lo = pos.floor() as usize;
            let frac = pos - lo as f64;
            let a = sorted[lo - 1] as f64;
            let b = sorted[lo] as f64;
            a + (b - a) * frac
        };
        Some([at(1), at(2), at(3)])
    }

    /// The chunk size for a file of `count` HSPs, or `None` for "do not
    /// partition". Mirrors the script's `chunk_size < 0` estimation branch,
    /// which is what `-D` selects (no size tuning on `-D` yet).
    pub fn chunk_size_for(&self, count: usize) -> Option<usize> {
        if count < MIN_CHUNK_SIZE {
            return None;
        }
        let chunk = if self.history.len() < 2 {
            // Too few prior outputs to estimate from; the script uses the cap.
            MAX_CHUNK_SIZE
        } else {
            let mut sorted = self.history.clone();
            sorted.sort_unstable();
            let q = Self::quantiles_exclusive(&sorted).expect("len >= 2 checked above");
            // <7 data points: outliers skew the estimate, so the script drops to
            // the median; at 7 or more it uses the upper quartile.
            let pick = if sorted.len() < 7 { q[1] } else { q[2] };
            (pick as usize).min(MAX_CHUNK_SIZE)
        };
        // "no need to sort if number of lines <= chunk_size"
        if count <= chunk { None } else { Some(chunk) }
    }

    /// Records a logical output size so later files can estimate from it.
    pub fn observe(&mut self, count: usize) {
        self.history.push(count);
    }

    /// Plans one logical output file. `strand` selects the diagonal key.
    pub fn plan(&mut self, recs: Vec<Record>, strand: char) -> Plan {
        let count = recs.len();
        self.observe(count);
        let Some(chunk) = self.chunk_size_for(count) else {
            return Plan::Whole(recs);
        };
        Plan::Split(split(recs, strand, chunk))
    }
}

/// Groups by `(reference chr, query chr)`, splits the busy pairs along their
/// diagonal, and bin-packs the rest. Never sorts or splits across a pair.
pub fn split(recs: Vec<Record>, strand: char, chunk: usize) -> Vec<Vec<Record>> {
    // Group in first-appearance order; that order defines both the split
    // sequence and the query-key order used to re-sort each aggregate.
    let mut pairs: Vec<((u32, u32), Vec<Record>)> = Vec::new();
    for r in recs {
        let key = (r.r_chr, r.q_chr);
        match pairs.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(r),
            None => pairs.push((key, vec![r])),
        }
    }

    // First appearance order of query keys, for the aggregate re-sort. lastz
    // requires query names in query-file order, which bin-packing can violate.
    let mut query_order: Vec<u32> = Vec::new();
    for ((_, q), _) in &pairs {
        if !query_order.contains(q) {
            query_order.push(*q);
        }
    }
    let query_rank = |q: u32| {
        query_order
            .iter()
            .position(|&x| x == q)
            .unwrap_or(usize::MAX)
    };

    // A single pair is always split: the script only builds skip_pairs when
    // there is more than one pair.
    let multi = pairs.len() > 1;
    let mut out: Vec<Vec<Record>> = Vec::new();
    let mut skip: Vec<((u32, u32), Vec<Record>)> = Vec::new();

    for (key, mut group) in pairs {
        if multi && group.len() <= chunk {
            skip.push((key, group));
            continue;
        }
        group.sort_by_key(|r| r.diagonal_key(strand));
        for piece in group.chunks(chunk) {
            out.push(piece.to_vec());
        }
    }

    if !skip.is_empty() {
        // Ascending by (count, pair), then greedy first-fit-by-order packing —
        // the script's `sorted([(len, pair)])` followed by its aggregation loop.
        skip.sort_by_key(|(k, v)| (v.len(), *k));
        // the index is the partition number being filled
        #[allow(clippy::needless_range_loop)]
        // the tuple is the partition key; naming it would not make it clearer
        #[allow(clippy::type_complexity)]
        let mut bins: Vec<Vec<((u32, u32), Vec<Record>)>> = vec![Vec::new()];
        let mut current = 0usize;
        for (key, group) in skip {
            if current + group.len() <= chunk {
                current += group.len();
            } else {
                bins.push(Vec::new());
                current = group.len();
            }
            bins.last_mut()
                .expect("a bin always exists")
                .push((key, group));
        }
        for mut bin in bins {
            if bin.is_empty() {
                continue;
            }
            bin.sort_by_key(|((_, q), _)| query_rank(*q));
            out.push(bin.into_iter().flat_map(|(_, v)| v).collect());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(rc: u32, qc: u32, rs: usize, qs: usize, len: usize) -> Record {
        Record {
            r_chr: rc,
            q_chr: qc,
            r_start: rs,
            r_end: rs + len,
            q_start: qs,
            q_end: qs + len,
            score: 100,
        }
    }

    #[test]
    fn plus_sorts_by_sum_and_minus_by_difference() {
        // Same midpoints, so only the key form distinguishes the two strands.
        let a = rec(0, 0, 100, 10, 0); // mids (100, 10): sum 110, diff -90
        let b = rec(0, 0, 10, 100, 0); // mids (10, 100): sum 110, diff  90
        let c = rec(0, 0, 1, 1, 0); //    mids (1, 1):   sum   2, diff   0
        assert_eq!(c.diagonal_key('+').0, 2);
        assert_eq!(a.diagonal_key('+').0, b.diagonal_key('+').0, "sum key ties");
        assert!(
            a.diagonal_key('-').0 < c.diagonal_key('-').0,
            "difference key orders"
        );
        assert!(c.diagonal_key('-').0 < b.diagonal_key('-').0);
    }

    #[test]
    fn midpoint_uses_the_reference_half_distance() {
        // len 7 -> half_dist 3, added to both starts.
        let r = rec(0, 0, 10, 100, 7);
        assert_eq!(r.diagonal_key('+'), (13 + 103, 13));
    }

    #[test]
    fn small_files_are_never_partitioned() {
        let p = Partitioner::default();
        assert_eq!(p.chunk_size_for(MIN_CHUNK_SIZE - 1), None);
    }

    #[test]
    fn without_history_the_cap_is_used_and_only_larger_files_split() {
        let p = Partitioner::default();
        assert_eq!(
            p.chunk_size_for(MAX_CHUNK_SIZE),
            None,
            "count <= chunk means whole"
        );
        assert_eq!(p.chunk_size_for(MAX_CHUNK_SIZE + 1), Some(MAX_CHUNK_SIZE));
    }

    #[test]
    fn quantiles_match_cpython_exclusive_method() {
        // statistics.quantiles([1,2,3,4]) == [1.25, 2.5, 3.75]
        let q = Partitioner::quantiles_exclusive(&[1, 2, 3, 4]).unwrap();
        assert!((q[0] - 1.25).abs() < 1e-9, "{q:?}");
        assert!((q[1] - 2.5).abs() < 1e-9, "{q:?}");
        assert!((q[2] - 3.75).abs() < 1e-9, "{q:?}");
    }

    #[test]
    fn a_single_pair_is_split_even_when_it_would_fit() {
        // multi == false, so the skip-pair path cannot apply.
        let recs: Vec<Record> = (0..10).map(|i| rec(0, 0, i * 10, i * 10, 5)).collect();
        let files = split(recs, '+', 4);
        assert_eq!(files.len(), 3, "10 records in chunks of 4");
        assert_eq!(files.iter().map(Vec::len).sum::<usize>(), 10);
    }

    #[test]
    fn small_pairs_are_aggregated_not_split_individually() {
        // Three pairs of 2 with chunk 4: one big pair splits, the two small ones
        // land together in one aggregate.
        let mut recs: Vec<Record> = (0..6).map(|i| rec(0, 0, i * 10, i * 10, 1)).collect();
        recs.extend((0..2).map(|i| rec(1, 1, i * 10, i * 10, 1)));
        recs.extend((0..2).map(|i| rec(2, 2, i * 10, i * 10, 1)));
        let files = split(recs, '+', 4);
        let total: usize = files.iter().map(Vec::len).sum();
        assert_eq!(total, 10, "no record lost or duplicated");
        // pair(0,0) has 6 > 4 so it splits into 4 + 2; the two 2-record pairs
        // aggregate into a single 4-record file.
        let mut sizes: Vec<usize> = files.iter().map(Vec::len).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![2, 4, 4], "{files:?}");
    }

    #[test]
    fn partition_union_preserves_the_record_multiset() {
        let mut recs: Vec<Record> = Vec::new();
        for chr in 0..3u32 {
            for i in 0..7 {
                recs.push(rec(chr, chr, i * 3, i * 5, 2));
            }
        }
        let mut before = recs.clone();
        let files = split(recs, '-', 3);
        let mut after: Vec<Record> = files.into_iter().flatten().collect();
        let key = |r: &Record| (r.r_chr, r.q_chr, r.r_start, r.q_start, r.score);
        before.sort_by_key(key);
        after.sort_by_key(key);
        assert_eq!(before, after, "union must equal the input multiset");
    }

    /// Execute the *actual* KegAlign `diagonal_partition.py`
    /// on the same records and require file-identical partition output. Skips
    /// silently when the oracle script or `bashlex` isn't available.
    #[test]
    fn partitions_match_the_oracle_script() {
        use crate::hsp::{Record as HspRecord, render_records};
        use crate::sequence::Chr;
        use std::process::Command;

        let root = std::env::var("HSPZ_ORACLE_ROOT").unwrap_or_else(|_| "/tmp/kegalign".into());
        let script = std::path::PathBuf::from(&root).join("scripts/diagonal_partition.py");
        let has_bashlex = Command::new("python3")
            .args(["-c", "import bashlex"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !script.is_file() || !has_bashlex {
            eprintln!("skip: oracle diagonal_partition.py or bashlex unavailable");
            return;
        }

        const CHUNK: usize = 2500;
        let r_chrs = vec![
            Chr {
                name: "refA".into(),
                start: 0,
                len: 100_000,
            },
            Chr {
                name: "refB".into(),
                start: 0,
                len: 100_000,
            },
        ];
        let q_chrs = vec![
            Chr {
                name: "qryA".into(),
                start: 0,
                len: 100_000,
            },
            Chr {
                name: "qryB".into(),
                start: 0,
                len: 100_000,
            },
        ];

        // One big pair (refA, qryA) across several diagonals, and one small
        // pair (refB, qryB) that must be aggregated, not split.
        let mut recs: Vec<HspRecord> = Vec::new();
        for i in 0..5500u32 {
            let rs = 100 + i as usize;
            let len = 50 + (i % 10) as usize;
            let qs = 50 + (i / 2) as usize;
            recs.push(HspRecord {
                r_chr: 0,
                q_chr: 0,
                r_start: rs,
                r_end: rs + len,
                q_start: qs,
                q_end: qs + len,
                score: 100,
            });
        }
        for i in 0..2000u32 {
            let rs = 100 + i as usize;
            let len = 40 + (i % 5) as usize;
            let qs = 50 + i as usize;
            recs.push(HspRecord {
                r_chr: 1,
                q_chr: 1,
                r_start: rs,
                r_end: rs + len,
                q_start: qs,
                q_end: qs + len,
                score: 100,
            });
        }

        let ours = split(recs.clone(), '-', CHUNK);

        let dir = std::env::temp_dir().join(format!("hspz-oracle-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("big.segments"),
            render_records(&recs, &r_chrs, &q_chrs, '-'),
        )
        .unwrap();

        // A fixed chunk size (2500 > 0) bypasses the script's estimation branch,
        // making its output deterministic.
        let out = Command::new("python3")
            .arg(&script)
            .arg(CHUNK.to_string())
            .args([
                "--strand=minus",
                "--segments=big.segments",
                "--output=big.segments",
                "big.err",
            ])
            .current_dir(&dir)
            .output()
            .expect("run oracle");
        assert!(
            out.status.success(),
            "oracle failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );

        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".split") && n.ends_with(".segments"))
            .collect();
        // Numeric, not lexicographic: `.split10` sorts before `.split2` as text,
        // which would silently misalign this comparison the moment the fixture
        // grows past 9 partitions.
        names.sort_by_key(|n| {
            n.split(".split")
                .nth(1)
                .and_then(|t| t.split('.').next())
                .and_then(|d| d.parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        });
        assert_eq!(
            names.len(),
            ours.len(),
            "oracle produced a different number of partitions"
        );
        for (i, name) in names.iter().enumerate() {
            let expected = render_records(&ours[i], &r_chrs, &q_chrs, '-');
            let got = std::fs::read_to_string(dir.join(name)).unwrap();
            assert_eq!(got, expected, "{name} differs from hspz split[{i}]");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
