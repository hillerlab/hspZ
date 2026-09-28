// Copyright (c) 2026 The Hiller Lab at the Senckenberg Gesellschaft für Naturforschung
// Distributed under the terms of the GNU General Public License, Version 3.0.

// Author : Alejandro Gonzales-Irribarren
// Github : alejandrogzi
// Email  : alejandrxgzi@gmail.com

//! S0 survivor audit — an env-gated diagnostic (`HSPZ_ANCHOR_CENSUS`), off the
//! timed path.
//!
//! The question S0 answers is whether the repeat tail is removable: of the
//! candidate hits that come from high-copy k-mer buckets, how many survive the
//! score gate, how many become accepted HSPs, and how strong those HSPs are.
//!
//! What it deliberately does **not** do is copy raw anchors. The previous
//! same-diagonal census appended every packed anchor to a host vector and then
//! sorted globally; on the pinned canonical fixture that is 137,916,664,712
//! anchors, about 1.10 TB of D2H traffic and the same again resident. That path
//! is not an admissible sampler and was deleted rather than capped, because a
//! bounded *final* reservoir does not bound the transfer. Raw-anchor pricing is
//! S1's problem and S1 is not written until S0 proves it is needed.
//!
//! Everything here folds into fixed-size scalars. Per seed batch S0 copies the
//! raw per-seed counts once; per launch it copies three active survivor-array
//! prefixes, folds them, and drops them. Peak resident is one batch plus one
//! launch and is reported so the bound can be checked rather than asserted.
//!
//! The accepted bit comes from `buf_done` in the window between `find_hsps` and
//! the done-scan, not from the `counters` feature: `counters` changes register
//! pressure and is banned from timed builds, and depending on it would mean S0
//! could not be verified byte-identical against its own env-off build.

use crate::hsp::SegmentPair;

/// Frozen r-bucket edges, identical to [`crate::gpu::HitStats`] so S0's numbers
/// are directly comparable with `--hit-stats` and with the r82 census. The
/// campaign's `r > 32` threshold is exactly the `5-32` / `33-256` boundary.
pub const LABELS: [&str; 6] = ["0", "1", "2-4", "5-32", "33-256", ">256"];

#[inline]
fn bucket(r: u32) -> usize {
    match r {
        0 => 0,
        1 => 1,
        2..=4 => 2,
        5..=32 => 3,
        33..=256 => 4,
        _ => 5,
    }
}

/// Accepted-HSP score band, so a bucket that survives with only weak HSPs can
/// be told from one carrying strong anchors. The frozen budget
/// protects a top LASTZ-score band, which S0 cannot compute; HSP score is the
/// available proxy and 10,000 is roughly 4x the campaign's K=2400.
const STRONG_HSP_SCORE: i32 = 10_000;

#[derive(Default, Clone, Copy)]
struct Bucket {
    seeds: u64,
    hits: u64,
    survivors: u64,
    accepted: u64,
    accepted_score: i64,
    accepted_strong: u64,
}

/// One entropy-accepted raw HSP with the multiplicity class of its seed.
/// Coordinates remain block-relative here; `run` maps them with the same
/// chromosome tables as production output before writing the diagnostic dump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcceptedHsp {
    pub hsp: SegmentPair,
    pub common: bool,
}

#[derive(Default)]
pub struct SurvivorAudit {
    buckets: [Bucket; 6],
    launches: u64,
    /// Launches whose survivor set could not be attributed, which must be zero
    /// for the report to mean anything.
    unattributed: u64,
    peak_launch_bytes: u64,
    total_bytes: u64,
}

impl SurvivorAudit {
    pub fn enabled() -> bool {
        std::env::var_os("HSPZ_ANCHOR_CENSUS").is_some()
    }

    /// A non-empty value other than `1` also requests the AL3 accepted-HSP
    /// dump. Bare `1` keeps the original scalar-only S0 diagnostic.
    pub fn dump_path() -> Option<std::path::PathBuf> {
        std::env::var_os("HSPZ_ANCHOR_CENSUS").and_then(|value| {
            if value.is_empty() || value == "1" { None } else { Some(value.into()) }
        })
    }

    /// Folds the raw hit counts copied once for one seed batch. Keeping this
    /// separate from [`ingest_survivors`](Self::ingest_survivors) prevents a
    /// `MAX_HITS` split from counting or copying the same seeds repeatedly.
    pub fn observe_counts(&mut self, counts: &[u32]) {
        for &r in counts {
            let b = &mut self.buckets[bucket(r)];
            b.seeds += 1;
            b.hits += r as u64;
        }
        let bytes = std::mem::size_of_val(counts) as u64;
        self.peak_launch_bytes = self.peak_launch_bytes.max(bytes);
        self.total_bytes += bytes;
    }

    /// Folds the score survivors of one `MAX_HITS` chunk.
    ///
    /// `counts` contains the raw per-seed multiplicities for this chunk.
    /// `survivors` are chunk-local hit IDs in ascending order, guaranteed by
    /// `emit_survivors`' stable prefix compaction, so attribution is one
    /// monotone walk. Accepted HSPs are appended for AL3/AM1; their coordinate
    /// conversion remains one layer up where chromosome tables are available.
    pub fn ingest_survivors(
        &mut self,
        counts: &[u32],
        survivors: &[u32],
        accepted: &[u32],
        hsps: &[SegmentPair],
        count_resident_bytes: u64,
        accepted_hsps: &mut Vec<AcceptedHsp>,
    ) {
        assert_eq!(survivors.len(), accepted.len());
        assert_eq!(survivors.len(), hsps.len());
        self.launches += 1;
        let bytes = (std::mem::size_of_val(survivors)
            + std::mem::size_of_val(accepted)
            + std::mem::size_of_val(hsps)) as u64;
        self.peak_launch_bytes = self.peak_launch_bytes.max(count_resident_bytes + bytes);
        self.total_bytes += bytes;

        let mut seed = 0usize;
        let mut end = counts.first().copied().unwrap_or(0);
        for (k, &id) in survivors.iter().enumerate() {
            while seed < counts.len() && end <= id {
                seed += 1;
                if let Some(&r) = counts.get(seed) {
                    end = end.saturating_add(r);
                }
            }
            if seed >= counts.len() {
                self.unattributed += (survivors.len() - k) as u64;
                break;
            }
            let r = counts[seed];
            let b = &mut self.buckets[bucket(r)];
            b.survivors += 1;
            if accepted[k] != 0 {
                b.accepted += 1;
                let hsp = hsps[k];
                let s = hsp.score;
                b.accepted_score += s as i64;
                if s >= STRONG_HSP_SCORE {
                    b.accepted_strong += 1;
                }
                accepted_hsps.push(AcceptedHsp { hsp, common: r > 32 });
            }
        }
    }

    /// Folds another engine's audit in. The report is emitted per reference bin,
    /// so without this the pass-level number would have to be summed by hand
    /// across bins — the same reason `HitStats` has a `merge`.
    pub fn merge(&mut self, other: &Self) {
        for (a, b) in self.buckets.iter_mut().zip(other.buckets.iter()) {
            a.seeds += b.seeds;
            a.hits += b.hits;
            a.survivors += b.survivors;
            a.accepted += b.accepted;
            a.accepted_score += b.accepted_score;
            a.accepted_strong += b.accepted_strong;
        }
        self.launches += other.launches;
        self.unattributed += other.unattributed;
        self.peak_launch_bytes = self.peak_launch_bytes.max(other.peak_launch_bytes);
        self.total_bytes += other.total_bytes;
    }

    pub fn report(&self) -> String {
        let tot = |f: fn(&Bucket) -> u64| -> u64 { self.buckets.iter().map(f).sum() };
        let (hits, surv, acc) = (tot(|b| b.hits), tot(|b| b.survivors), tot(|b| b.accepted));
        let pct = |n: u64, d: u64| if d == 0 { 0.0 } else { n as f64 / d as f64 * 100.0 };

        let mut out = String::from(
            "S0 SURVIVOR AUDIT  (HSPZ_ANCHOR_CENSUS)\n  \
             r bucket        seeds            hits    hit%      survivors   surv%   \
             accepted    acc%   mean score   strong\n",
        );
        for (i, l) in LABELS.iter().enumerate() {
            let b = &self.buckets[i];
            out.push_str(&format!(
                "  {l:<10} {:>12} {:>15} {:>7.2}% {:>14} {:>7.2}% {:>10} {:>7.2}% {:>12.0} {:>8}\n",
                b.seeds,
                b.hits,
                pct(b.hits, hits),
                b.survivors,
                pct(b.survivors, surv),
                b.accepted,
                pct(b.accepted, acc),
                if b.accepted == 0 { 0.0 } else { b.accepted_score as f64 / b.accepted as f64 },
                b.accepted_strong,
            ));
        }

        // The two decision numbers, at the frozen r>32 threshold.
        let common = |f: fn(&Bucket) -> u64| f(&self.buckets[4]) + f(&self.buckets[5]);
        out.push_str(&format!(
            "  ---\n  r>32 (common) share:  hits {:.2}%   survivors {:.2}%   accepted HSPs {:.2}%   \
             strong accepted {:.2}%\n",
            pct(common(|b| b.hits), hits),
            pct(common(|b| b.survivors), surv),
            pct(common(|b| b.accepted), acc),
            pct(common(|b| b.accepted_strong), tot(|b| b.accepted_strong)),
        ));
        out.push_str(&format!(
            "  launches {}  unattributed survivors {}  D2H peak/launch {:.2} MiB  total {:.2} GiB\n",
            self.launches,
            self.unattributed,
            self.peak_launch_bytes as f64 / (1 << 20) as f64,
            self.total_bytes as f64 / (1u64 << 30) as f64,
        ));
        // Review 10 AH3: this screen is one-directional.
        out.push_str(
            "  reading: ~0% common survivors licenses rank 2; a non-zero share is an UPPER\n  \
             bound on raw-HSP damage only, never a rejection.\n",
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three seeds with r = 1, 40, 3 over one launch starting at hit 0. The
    /// middle seed is the only common one, and the join must attribute its
    /// survivors to the 33-256 bucket and nobody else's.
    #[test]
    fn monotone_join_attributes_survivors_to_the_owning_seed() {
        let counts = [1u32, 40, 3];
        // hit IDs: seed0 -> {0}, seed1 -> 1..=40, seed2 -> 41..=43
        let survivors = [0u32, 5, 40, 43];
        let accepted = [1u32, 1, 0, 1];
        let hsps = [
            SegmentPair { score: 50_000, ..Default::default() },
            SegmentPair { score: 3_000, ..Default::default() },
            SegmentPair::default(),
            SegmentPair { score: 12_000, ..Default::default() },
        ];
        let mut accepted_hsps = Vec::new();
        let mut a = SurvivorAudit::default();
        a.observe_counts(&counts);
        a.ingest_survivors(
            &counts,
            &survivors,
            &accepted,
            &hsps,
            std::mem::size_of_val(&counts) as u64,
            &mut accepted_hsps,
        );

        assert_eq!(a.unattributed, 0, "every survivor must land on a seed");
        assert_eq!(a.buckets[1].seeds, 1, "r=1 seed");
        assert_eq!(a.buckets[2].seeds, 1, "r=3 seed lands in 2-4");
        assert_eq!(a.buckets[4].seeds, 1, "r=40 seed lands in 33-256");
        assert_eq!(a.buckets[4].hits, 40);

        assert_eq!(a.buckets[1].survivors, 1);
        assert_eq!(a.buckets[4].survivors, 2, "ids 5 and 40 belong to the r=40 seed");
        assert_eq!(a.buckets[2].survivors, 1, "id 43 belongs to the r=3 seed");

        // Accepted and score bands follow the same attribution.
        assert_eq!(a.buckets[1].accepted, 1);
        assert_eq!(a.buckets[1].accepted_strong, 1, "50000 >= 10000");
        assert_eq!(a.buckets[4].accepted, 1, "only id 5 was accepted");
        assert_eq!(a.buckets[4].accepted_strong, 0, "3000 < 10000");
        assert_eq!(a.buckets[2].accepted_strong, 1);
        assert_eq!(accepted_hsps.len(), 3);
        assert!(!accepted_hsps[0].common);
        assert!(accepted_hsps[1].common);
        assert!(!accepted_hsps[2].common);
    }

    /// Survivor IDs restart at zero for every MAX_HITS chunk.
    #[test]
    fn chunk_local_ids_use_the_chunk_count_slice() {
        let counts = [10u32, 40];
        let survivors = [0u32, 9, 10]; // chunk-local: 0..=9 -> seed0, 10.. -> seed1
        let accepted = [0u32, 0, 0];
        let hsps = [SegmentPair::default(); 3];
        let mut out = Vec::new();
        let mut a = SurvivorAudit::default();
        a.observe_counts(&counts);
        a.ingest_survivors(&counts, &survivors, &accepted, &hsps, 8, &mut out);
        assert_eq!(a.unattributed, 0);
        assert_eq!(a.buckets[3].survivors, 2, "r=10 seed owns local ids 0..=9");
        assert_eq!(a.buckets[4].survivors, 1, "r=40 seed owns local id 10");
    }

    /// Per-bin reports must be summable, or the pass-level share is wrong.
    #[test]
    fn merge_sums_counts_and_keeps_peak() {
        let mut a = SurvivorAudit::default();
        a.observe_counts(&[40]);
        a.ingest_survivors(
            &[40],
            &[0],
            &[1],
            &[SegmentPair { score: 5_000, ..Default::default() }],
            4,
            &mut Vec::new(),
        );
        let mut b = SurvivorAudit::default();
        b.observe_counts(&[40, 40]);
        b.ingest_survivors(
            &[40, 40],
            &[0, 41],
            &[1, 1],
            &[
                SegmentPair { score: 20_000, ..Default::default() },
                SegmentPair { score: 20_000, ..Default::default() },
            ],
            8,
            &mut Vec::new(),
        );
        let peak_b = b.peak_launch_bytes;
        a.merge(&b);
        assert_eq!(a.buckets[4].seeds, 3, "1 + 2 seeds all in 33-256");
        assert_eq!(a.buckets[4].hits, 120);
        assert_eq!(a.buckets[4].survivors, 3);
        assert_eq!(a.buckets[4].accepted, 3);
        assert_eq!(a.buckets[4].accepted_strong, 2, "only the 20000s are strong");
        assert_eq!(a.launches, 2);
        assert_eq!(a.peak_launch_bytes, peak_b, "peak is a max, not a sum");
    }

    /// The transport bound must be observable, since that is the whole reason
    /// the raw-anchor census was deleted.
    #[test]
    fn transport_is_measured_and_reported() {
        let mut a = SurvivorAudit::default();
        let counts = [4u32];
        let hsps = [SegmentPair { score: 2_500, ..Default::default() }; 2];
        a.observe_counts(&counts);
        a.ingest_survivors(&counts, &[0, 1], &[1, 1], &hsps, 4, &mut Vec::new());
        let r = a.report();
        assert!(r.contains("D2H peak/launch"), "{r}");
        assert!(r.contains("unattributed survivors 0"), "{r}");
        assert_eq!(a.peak_launch_bytes, 4 + 2 * (4 + 4 + 16));
        assert_eq!(a.total_bytes, 4 + 2 * (4 + 4 + 16));
    }
}
